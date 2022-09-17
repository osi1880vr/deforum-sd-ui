class interpolator:


    def render_interpolation(self, args, anim_args):
        # animations use key framed prompts
        args.prompts = animation_prompts

        # create output folder for the batch
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving animation frames to {args.outdir}")

        # save settings for the batch
        settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

        # Interpolation Settings
        args.n_samples = 1
        args.seed_behavior = 'fixed'  # force fix seed at the moment bc only 1 seed is available
        prompts_c_s = []  # cache all the text embeddings

        print(f"Preparing for interpolation of the following...")

        for i, prompt in animation_prompts.items():
            args.prompt = prompt

            # sample the diffusion model
            results = generate(args, return_c=True)
            c, image = results[0], results[1]
            prompts_c_s.append(c)

            # display.clear_output(wait=True)
            display.display(image)

            args.seed = next_seed(args)

        display.clear_output(wait=True)
        print(f"Interpolation start...")

        frame_idx = 0

        if anim_args.interpolate_key_frames:
            for i in range(len(prompts_c_s) - 1):
                dist_frames = list(animation_prompts.items())[i + 1][0] - list(animation_prompts.items())[i][0]
                if dist_frames <= 0:
                    print("key frames duplicated or reversed. interpolation skipped.")
                    return
                else:
                    for j in range(dist_frames):
                        # interpolate the text embedding
                        prompt1_c = prompts_c_s[i]
                        prompt2_c = prompts_c_s[i + 1]
                        args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1 / dist_frames))

                        # sample the diffusion model
                        results = generate(args)
                        image = results[0]

                        filename = f"{args.timestring}_{frame_idx:05}.png"
                        image.save(os.path.join(args.outdir, filename))
                        frame_idx += 1

                        display.clear_output(wait=True)
                        display.display(image)

                        args.seed = next_seed(args)

        else:
            for i in range(len(prompts_c_s) - 1):
                for j in range(anim_args.interpolate_x_frames + 1):
                    # interpolate the text embedding
                    prompt1_c = prompts_c_s[i]
                    prompt2_c = prompts_c_s[i + 1]
                    args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1 / (anim_args.interpolate_x_frames + 1)))

                    # sample the diffusion model
                    results = generate(args)
                    image = results[0]

                    filename = f"{args.timestring}_{frame_idx:05}.png"
                    image.save(os.path.join(args.outdir, filename))
                    frame_idx += 1

                    display.clear_output(wait=True)
                    display.display(image)

                    args.seed = next_seed(args)

        # generate the last prompt
        args.init_c = prompts_c_s[-1]
        results = generate(args)
        image = results[0]
        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))

        display.clear_output(wait=True)
        display.display(image)
        args.seed = next_seed(args)

        # clear init_c
        args.init_c = None
