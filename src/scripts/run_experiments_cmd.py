import os
import re
import sys
import argparse
import subprocess

test_scenes = [
    ["common/scenes/bathroom/scene-v4.pbrt"],
    ["common/scenes/bedroom/scene-v4.pbrt"],
    ["common/scenes/living-room-2/scene-v4.pbrt"],
    ["common/scenes/staircase2/scene-v4.pbrt"],
    ["common/scenes/staircase/scene-v4.pbrt"],
    ["common/scenes/zero-day/frame25.pbrt"],
    ["common/scenes/bistro/bistro_cafe.pbrt"],
    
    # VPL (You need to set VPL_USED to true in src/pbrt/lightsamplers_constants.h before running the following scenes)
    # ["common/scenes/sanmiguel/sanmiguel-courtyard-second-vpl-vfinal.pbrt"],
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=str, default=".\\build\\Release\\pbrt.exe", help="Path to the executable")
    parser.add_argument("--output_dir", type=str, default=".\\common\\outputs", help="Base output directory")
    parser.add_argument("--lightsampler", type=str, default="bvh", help="Method name(s), separated by commas if multiple (e.g. 'bvh,random')")
    parser.add_argument("--budget_type", type=str, default="spp", choices=["spp", "time"], help="Type of training budget")
    parser.add_argument("--budget_value", type=float, default=256, help="Number of seconds or samples for training budget")
    parser.add_argument("--max_depth", type=int, default=1, help="max depth used for rendering")
    parser.add_argument("--train_budget_ratio", type=str, default="0.15", help="training budget ratio(s), comma separated if multiple (e.g. '0.15,0.2')")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs for each scene")
    parser.add_argument("--experiment_name", type=str, default="sample_exp", help="Experiment name")
    parser.add_argument("--intermediate_output_frequency", type=float, default=-1, help="Frequency of outputting intermediate results")
    parser.add_argument("--resolution", type=str, default="-1", help="Overwrite scene's resolution of the output image (e.g. '1280x720')")
    parser.add_argument("--show_display_server", action="store_true", help="Show display server while rendering")
    parser.add_argument("--log_verbose", action="store_true", help="Write verbose log")
    args = parser.parse_args()

    # Parse lightsamplers (e.g. "bvh,random")
    lightsamplers = [ls.strip() for ls in args.lightsampler.split(',') if ls.strip()]
    # Parse train_budget_ratio values as floats (e.g. "0.15,0.2" becomes [0.15, 0.2])
    train_budget_ratios = [float(r.strip()) for r in args.train_budget_ratio.split(',') if r.strip()]

    # Patterns for substitutions in the scene file.
    maxdepth_pattern = r'("integer maxdepth"\s*\[)[^\]]+(\])'
    lightsampler_pattern = r'("string lightsampler"\s*")[^"]*(")'
    trainingBudgetRatio_pattern = r'("float trainingBudgetRatio"\s*\[)[^\]]+(\])'
    xres_pattern = r'("integer xresolution"\s*\[\s*)\d+(\s*\])'
    yres_pattern = r'("integer yresolution"\s*\[\s*)\d+(\s*\])'

    if args.resolution != "-1":
        try:
            width_str, height_str = args.resolution.lower().split('x')
            width = int(width_str)
            height = int(height_str)
        except Exception as e:
            print("Invalid resolution format. Please use the format 'WIDTHxHEIGHT' (e.g. '1280x720').")
            sys.exit(1)
        # Regex patterns to match the xresolution and yresolution lines
        xres_repl = r'\g<1>{}\g<2>'.format(width)
        yres_repl = r'\g<1>{}\g<2>'.format(height)

    maxdepth_repl = r'\g<1>{}\g<2>'.format(args.max_depth)

    # Outer loops: iterate over each lightsampler and each train_budget_ratio.
    for lightsampler in lightsamplers:
        print("Processing lightsampler: {}".format(lightsampler))
        for train_budget_ratio in train_budget_ratios:
            print("Processing train_budget_ratio: {}".format(train_budget_ratio))
            for config_scene_file in test_scenes:
                with open(config_scene_file[0], 'r') as f:
                    scene_config = f.read()

                if args.resolution != "-1":
                    scene_config = re.sub(xres_pattern, xres_repl, scene_config)
                    scene_config = re.sub(yres_pattern, yres_repl, scene_config)

                # Apply substitutions:
                scene_config = re.sub(maxdepth_pattern, maxdepth_repl, scene_config)
                lightsampler_repl = r'\g<1>{}\g<2>'.format(lightsampler)
                scene_config = re.sub(lightsampler_pattern, lightsampler_repl, scene_config)
                trainingBudgetRatio_repl = r'\g<1>{}\g<2>'.format(train_budget_ratio)
                scene_config = re.sub(trainingBudgetRatio_pattern, trainingBudgetRatio_repl, scene_config)

                new_lines = []

                # If budget type is time, update or insert the timeLimit line.
                if args.budget_type == "time":
                    timeLimit_pattern = r'("float timeLimit"\s*\[)[^\]]+(\])'
                    timeLimit_repl = r'\1{}\2'.format(args.budget_value)
                    if not re.search(timeLimit_pattern, scene_config):
                        new_lines.append('"float timeLimit" [{}]'.format(args.budget_value))
                    else:
                        scene_config = re.sub(timeLimit_pattern, timeLimit_repl, scene_config)

                # Handle intermediate_output_frequency.
                if args.intermediate_output_frequency > 0:
                    if args.budget_type == "time":
                        interval_pattern = r'("float timeInterval"\s*\[)[^\]]+(\])'
                        interval_repl = r'\1{}\2'.format(args.intermediate_output_frequency)
                        interval_line = '"float timeInterval" [{}]'.format(args.intermediate_output_frequency)
                    else:
                        interval_pattern = r'("integer sppInterval"\s*\[)[^\]]+(\])'
                        interval_repl = r'\1{}\2'.format(int(args.intermediate_output_frequency))
                        interval_line = '"integer sppInterval" [{}]'.format(int(args.intermediate_output_frequency))
                    if not re.search(interval_pattern, scene_config):
                        new_lines.append(interval_line)
                    else:
                        scene_config = re.sub(interval_pattern, interval_repl, scene_config)

                if new_lines:
                    match = re.search(trainingBudgetRatio_pattern, scene_config)
                    if match:
                        end_pos = match.end()
                        scene_config = scene_config[:end_pos] + '\n' + '\n'.join(new_lines) + scene_config[end_pos:]
                    else:
                        print("Error: trainingBudgetRatio not found in scene file.")
                        sys.exit(1)

                dir_scene = os.path.dirname(config_scene_file[0])
                for run_i in range(args.n_runs):
                    # Replace the decimal point in train_budget_ratio to avoid issues in filenames
                    ratio_str = str(train_budget_ratio).replace('.', 'p')
                    tmp_config_method = os.path.join(dir_scene, "{}_{}_{}_{}.pbrt".format(lightsampler, args.experiment_name, ratio_str, run_i))
                    with open(tmp_config_method, "w") as f:
                        f.write(scene_config)

                    scene_name = os.path.splitext(os.path.basename(dir_scene))[0]
                    print("Testing scene: {} at run {} using lightsampler: {} and train_budget_ratio: {}".format(
                        scene_name, run_i, lightsampler, train_budget_ratio))
                    output_dir = os.path.join(args.output_dir, scene_name, "{}_{}_{}{}_{}b_{}trainratio_{}".format(
                        lightsampler, args.experiment_name, args.budget_value, args.budget_type,
                        args.max_depth, train_budget_ratio, run_i))
                    os.makedirs(output_dir, exist_ok=True)
                    output_img_path = os.path.join(output_dir, "render.exr")
                    output_consoleout_path = os.path.join(output_dir, "consoleout.log")

                    command_args = [args.exe, tmp_config_method, "-outfile", output_img_path, "--gpu", "--stats", "--seed", f"{run_i}"]
                    if args.budget_type == "spp":
                        command_args += ["--spp", str(int(args.budget_value))]
                    if args.show_display_server:
                        command_args += ["--display-server", "localhost:14158"]
                    if args.log_verbose:
                        output_log_path = os.path.join(output_dir, "verbose.log")
                        command_args += ["--logfile", output_log_path, "--log-level", "verbose"]

                    output = subprocess.run(command_args, capture_output=True)
                    with open(output_consoleout_path, "w") as f:
                        f.write(output.stdout.decode("utf-8"))

                    if output.returncode != 0:
                        print("Error: {}".format(output.stderr))
                        sys.exit(1)

                    print("\n>>>>>>>>>>>>>Done rendering image for: {}-{} using lightsampler: {} and train_budget_ratio: {}".format(
                        scene_name, run_i, lightsampler, train_budget_ratio))
                    os.remove(tmp_config_method)

    print("Done rendering all scenes.\nDeleting tmp files...", end=" ")
    print("Done.")
