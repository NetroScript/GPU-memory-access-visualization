import { svelte } from '@sveltejs/vite-plugin-svelte'
import type { UserConfig } from 'vite';
import {viteSingleFile} from "vite-plugin-singlefile";

const config: UserConfig = {
	plugins: [
		svelte(),
		viteSingleFile({
			removeViteModuleLoader: true,
			useRecommendedBuildConfig: true,
		}),
	],
	build: {
		outDir: 'build',
		target: 'esnext',
	},
};

export default config;
