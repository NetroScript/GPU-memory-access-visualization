import { svelte } from '@sveltejs/vite-plugin-svelte'
import type { UserConfig } from 'vite';
import {viteSingleFile} from "vite-plugin-singlefile";
import { vitePostBuildFix } from "./tools/postBuild";

const config: UserConfig = {
	plugins: [
		svelte(),
		viteSingleFile({
			removeViteModuleLoader: true,
			useRecommendedBuildConfig: true,
		}),
		vitePostBuildFix(),
	],
	build: {
		outDir: 'build',
		target: 'esnext',
	},
};

export default config;
