import os
import argparse
import torch
from booth_evaluation.personalized import PersonalizedBase
from booth_evaluation.clip_eval import CLIPEvaluator
from booth_evaluation.dino_eval import DINOEvaluator
from booth_evaluation.lpips_eval import LPIPSEvaluator
from statistics import mean

def run_evaluation(opt):
    object_name = opt.src_dir.split("/")[-1]
    live_objects = ["black_cat", "brown_cat", "brown_dog2", "brown_dog", "fat_dog"]
    if opt.mode == "challenging":
        if object_name in live_objects:
            with open("./inputs/prompts_live_objects_challenging.txt") as file:
                prompts = [line.rstrip() for line in file]
        else:
            with open("./inputs/prompts_nonlive_objects_challenging.txt") as file:
                prompts = [line.rstrip() for line in file]
    else:
        if object_name in live_objects:
            with open("./inputs/prompts_live_objects.txt") as file:
                prompts = [line.rstrip() for line in file]
        else:
            with open("./inputs/prompts_nonlive_objects.txt") as file:
                prompts = [line.rstrip() for line in file]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    src_data_loader = PersonalizedBase(opt.src_dir, size=256, flip_p=0.0, set="eval")
    img_sims_clip = []
    img_sims_dino = []
    img_divs_lpips = []
    img_pres_dino = []
    txt_sims = []
    clip_evaluator = CLIPEvaluator(device)
    lpips_evaluator = LPIPSEvaluator(device)
    dino_evaluator = DINOEvaluator(device)

    for i, prompt in enumerate(prompts):
        print(f"evaluating prompt {i}: {prompt}")

        trg_dir = os.path.join(opt.trg_dir, f"prompt_{i:02d}")
        if os.path.exists(os.path.join(trg_dir, "samples")):
            trg_dir = os.path.join(trg_dir, "samples")
        trg_data_loader = PersonalizedBase(trg_dir, size=256, flip_p=0.0)
        if opt.cls_dir is not None:
            cls_data_loader = PersonalizedBase(opt.cls_dir, size=256, flip_p=0.0)
        else:
            cls_data_loader = None

        src_images = [
            torch.from_numpy(src_data_loader[i]["image"]).permute(2, 0, 1)
            for i in range(src_data_loader.num_images)
        ]
        trg_images = [
            torch.from_numpy(trg_data_loader[i]["image"]).permute(2, 0, 1)
            for i in range(trg_data_loader.num_images)
        ]
        src_images = torch.stack(src_images, axis=0)
        trg_images = torch.stack(trg_images, axis=0)
        if cls_data_loader is not None:
            cls_images = [
                torch.from_numpy(cls_data_loader[i]["image"]).permute(2, 0, 1)
                for i in range(cls_data_loader.num_images)
            ]
            cls_images = torch.stack(cls_images, axis=0)
            pre_img_dino = dino_evaluator.img_to_img_similarity(src_images, cls_images)
            pre_img_dino = float(pre_img_dino.cpu().numpy())
            img_pres_dino.append(pre_img_dino)
        else:
            cls_images = None
            img_pres_dino = None

        sim_img_dino = dino_evaluator.img_to_img_similarity(src_images, trg_images)
        sim_img_clip = clip_evaluator.img_to_img_similarity(src_images, trg_images)
        div_img_lpips = lpips_evaluator.compute_pairwise_distance(trg_images)
        sim_text = clip_evaluator.txt_to_img_similarity(prompt, trg_images)

        sim_img_clip = float(sim_img_clip.cpu().numpy())
        sim_img_dino = float(sim_img_dino.cpu().numpy())
        sim_text = float(sim_text.cpu().numpy())

        img_sims_clip.append(sim_img_clip)
        img_sims_dino.append(sim_img_dino)
        img_divs_lpips.append(div_img_lpips)
        txt_sims.append(sim_text)

    metrics_file = f"{opt.trg_dir}/{opt.concept}_metrics.txt"
    with open(metrics_file, "w") as f:
        if img_pres_dino is not None:
            for prompt, sim_dino, sim_clip, sim_text, div_lpips, pre_dino in zip(prompts, img_sims_dino, img_sims_clip, txt_sims, img_divs_lpips, img_pres_dino):
                f.write(f"{pre_dino}\t{div_lpips}\t{sim_dino}\t{sim_clip}\t{sim_text}\t{prompt}\n")
        else:
            for prompt, sim_dino, sim_clip, sim_text, div_lpips in zip(prompts, img_sims_dino, img_sims_clip, txt_sims, img_divs_lpips):
                f.write(f"{div_lpips}\t{sim_dino}\t{sim_clip}\t{sim_text}\t{prompt}\n")

    if img_pres_dino is not None:
        avg_img_dino_pres = mean(img_pres_dino)
    else:
        avg_img_dino_pres = None
    avg_img_dino = mean(img_sims_dino)
    avg_img_clip = mean(img_sims_clip)
    avg_img_lpips = mean(img_divs_lpips)
    avg_txt_clip = mean(txt_sims)

    summary_file = f"{opt.trg_dir}/metrics.txt"
    with open(summary_file, "w") as f:
        if img_pres_dino is not None:
            f.write(f"\n{opt.concept}\t{avg_img_dino_pres}\t{avg_img_lpips}\t{avg_img_dino}\t{avg_img_clip}\t{avg_txt_clip}")
        else:
            f.write(f"\n{opt.concept}\t{avg_img_lpips}\t{avg_img_dino}\t{avg_img_clip}\t{avg_txt_clip}")

    print("total dino img sim: ", avg_img_dino)
    print("total clip img sim: ", avg_img_clip)
    print("total lpips img div: ", avg_img_lpips)
    print("total txt sim: ", avg_txt_clip)
    print(f"metrics were saved in {summary_file}")

    return avg_img_dino, avg_img_clip, avg_txt_clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, help="dir to result images")
    parser.add_argument("--cls_result_dir", type=str, default="", help="dir to result images")
    parser.add_argument("--mode", type=str, help="mode to evaluate", default="normal", choices=["challenging", "normal"])
    args = parser.parse_args()

    vico_concepts = {
        "fat_dog": "dog",
        "brown_dog": "dog",
        "brown_dog2": "dog",
        "black_cat": "cat",
        "brown_cat": "cat",
        ####################
        "cat_statue": "statue",
        "elephant_statue": "statue",
        "duck_toy": "toy",
        "monster_toy": "toy",
        "brown_teddybear": "teddybear",
        ####################
        "tortoise_plushy": "plushy",
        "alarm_clock": "clock",
        "pink_sunglasses": "sunglasses",
        "red_teapot": "teapot",
        "red_vase": "vase",
        "wooden_barn": "barn",
        ##################
    }

    concept_results = []
    for concept_dir in os.listdir(args.result_dir):
        concept_path = os.path.join(args.result_dir, concept_dir)
        if not os.path.isdir(concept_path):
            continue

        concept = concept_dir.split("-")[0]
        if concept_dir not in vico_concepts:
            print(f"Skipping {concept_dir} as it's not in vico_concepts")
            continue

        print(f"Processing {concept_dir}")
        args.concept = concept
        # args.src_dir = f"./inputs/{vico_concepts[concept_dir]}"
        args.src_dir = f"./inputs/{concept_dir}"
        args.trg_dir = concept_path
        if len(args.cls_result_dir) > 0:
            args.cls_dir = os.path.join(args.cls_result_dir, concept_dir)
        else:
            args.cls_dir = None

        avg_dino, avg_clip, avg_txt = run_evaluation(args)
        concept_results.append((avg_dino, avg_clip, avg_txt))

    if concept_results:
        overall_avg_dino = mean([r[0] for r in concept_results])
        overall_avg_clip = mean([r[1] for r in concept_results])
        overall_avg_txt = mean([r[2] for r in concept_results])
    else:
        overall_avg_dino = overall_avg_clip = overall_avg_txt = 0.0

    overall_file = os.path.join(args.result_dir, "overall_metrics.txt")
    with open(overall_file, "w") as f:
        f.write(f"{args.result_dir} {overall_avg_dino:.5f} {overall_avg_clip:.5f} {overall_avg_txt:.5f}\n")
    print(f"Overall averaged metrics saved in {overall_file}")
