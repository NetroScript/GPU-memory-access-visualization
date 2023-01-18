# GPU-memory-access-visualization

This folders contains the project which generates the HTML template.

It is build on the following technologies (those are the most important ones):

* TypeScript
* [Svelte](https://svelte.dev/)
* [Skeleton UI Framework](https://skeleton.dev)
* [Tailwind CSS](https://tailwindcss.com/)
* [Vite](https://vitejs.dev/)

More information about the packages and their uses in the section below.

But first how to use and install the project.

### Table of Content

<!-- TOC -->
* [GPU-memory-access-visualization](#gpu-memory-access-visualization)
    * [Table of Content](#table-of-content)
  * [Using the project](#using-the-project)
  * [Installing the project](#installing-the-project)
  * [Developing](#developing)
  * [Building](#building)
  * [More details about the used packages and their files](#more-details-about-the-used-packages-and-their-files)
  * [Entrypoint for Development](#entrypoint-for-development)
<!-- TOC -->

## Using the project

To use the project, you don't need to do anything else. There should be a pre-built HTML file ready for you to use. You can use this file as a template for the C++ header (when using the `parseDataForJSPage` generator).

If you need to rebuild or make changes to the project, you will have to install the dependencies and then run the described commands.

## Installing the project

For this project you need NodeJS and NPM installed. You can get them from [here](https://nodejs.org/en/).
The latest (LTS) version should be fine.

Both node and npm need to be on the path.

After that, you should install pnpm, as this is the package manager used in this project.
[Note: You can also use npm, but then you need to change the commands in the following sections and you won't be able to use the pnpm lockfile]

You can install pnpm with the following command:

```bash
# Install pnpm
npm install -g pnpm
```

Now that you installed pnpm, you need to install all the dependencies for the project.

You can do that with the following command:

```bash
# Install all dependencies
pnpm install
```

And you are already done with the installation. You can now modify the project and build it.

## Developing

During Development you can use the following command to start a development server:

```bash
pnpm dev
# or start the server and open the app in a new browser tab
pnpm dev -- --open
```

This development server will automatically rebuild the project when you change a file. 
So you can instantly see changes in the browser.

You can see all available commands in the `package.json` file.

## Building

To bundle the entire application into the single HTML template, you can use the following command:

```bash
pnpm build
```

This will currently create a `build` folder with an `index.html` file in it.

## More details about the used packages and their files

All packages used in `package.json` are listed here with a short explanation (in alphabetical order).

* `@iconify/icons-ic`
  * Add the `icons-ic` icon set for offline use. A list of all available icons can be found [here](https://icon-sets.iconify.design/ic/).
* `@iconify/svelte`
  * Add a icon component of the `iconify` library which is usable in Svelte.
* `@skeletonlabs/skeleton`
  * This is a UI framework which provides some pre-made components for svelte (like the drawer) and additionally dictates the default style, the official docs can be found [here](https://www.skeleton.dev/docs/why)
* `@sveltejs/vite-plugin-svelte`
  * This allows to integrate Svelte easily into Vite. It is applied inside `vite.config.ts` as the first used plugin. At the same time, it forwards Vite's preprocessor to the Svelte config in `svelte.config.js`. This allows Svelte to have for example style blocks which use the `PostCSS` language or Script blocks which use TypeScript.
* `@tailwindcss/forms`
  * A plugin for tailwindcss which makes it easier to work with forms. (By resetting the default browser form styles and also making them available as classes for other elements)
* `@tailwindcss/line-clamp`
  * A utility which allows to clamp text after multiple lines. (By default in CSS you can only clamp a single line)
* `@tailwindcss/typography`
  * Provides a `prose` class which automatically styles all elements within this container with respect to their tags (like `h1`, `h2`, `span`, ...) as by default the style of those classes is removed.
* `@tsconfig/svelte`
  * This provides a default `.tsconfig` for Svelte projects and can be inherited for the own `tsconfig`.
* `@typescript-eslint/eslint-plugin` + `@typescript-eslint/parser`
  * Add ESLint (rule) support to TypeScript code.
* `autoprefixer`
  * A PostCSS plugin to automatically add necessary browser prefixes to CSS, so that more browsers are supported.
* `color2k`
  * A compact library for color manipulations. (Interpolate between colors, make them more transparent, and be able to output this as a color for CSS)
* `esbuild`
  * Used by Vite for minification and transpilation. (Also supports features like tree shaking)
* `eslint`
  * Static Code analyzer to show code errors (or warnings). ESLint can automatically fix certain errors or guidelines. In this project the configuration is mostly managed by Prettier.
  * The rules for ESLint can be found in `.eslintrc.cjs`. Ignored files for ESLint are in `.eslintignore`. 
* `eslint-config-prettier`
  * Makes ESLint compatible with Prettier.
* `eslint-plugin-svelte3`
  * Makes ESLint compatible with Svelte.
* `postcss`
  * Post-processes CSS files to allow plugins to work with them (like `autoprefixer`). For example transforms newer syntax into browser-compatible CSS code and automatically generates polyfills among other things.
* `prettier`
  * Opinionated Code Formatter (more or less ESLint with pre-made rules). 
  * The configuration file for the project can be found in `.prettierrc`, ignored files are defined in `.prettierignore`.
* `prettier-plugin-svelte`
  * This makes prettier compatible with `.svelte` files.
* `rollup`
  * Used by Vite for bundling files and Vite plugins extend Rollup's plugin interface.
* `svelte`
  * The used Component framework for the application. Components made with Svelte end in `.svelte`. An introduction tutorial is available [here](https://svelte.dev/tutorial/basics), the overall docs [here](https://svelte.dev/docs).
  * The configuration can be found at `svelte.config.js`.
* `svelte-check`
  * A command-line tool (also displayed during build) which checks for type errors, unused CSS and accessibility for `.svelte` files.
* `tailwindcss`
  * A CSS framework focused on utility classes which speeds up UI development. It provides classes for relatively basic CSS functions allowing you to compose these into blocks you need. Docs are available [here](https://tailwindcss.com/docs/). They might be necessairy to get previews of for example colors or widths and distances.
  * The configuration and loaded plugins can be found at `tailwind.config.cjs`.
* `tslib`
  * Runtime library for TypeScript which contains commonly used TypeScript helper functions. (Which reduces duplicate code on export)
* `typescript`
  * Module which allows the usage (and compilation) of TypeScript, which is a superset of JavaScript, which adds static typing to JavaScript which thens allows extensive static analysis of the code.
  * The configuration for the project can be found at `tsconfig.json`.
* `vite`
  * A frontend build tooling, which allows to have a HMR (Hot Module Replacement) development server, and to then also bundle all necessary files into an exported application.
  * The configuration is in `vite.config.ts`
* `vite-plugin-singlefile`
  * A vite plugin which allows bundling all outputs files into a single HTML file. (Instead of for example 1 big HTML with 1 CSS and 1 JS file)

## Entrypoint for Development

This section will shortly describe where you can find the individual components of the project and where you would need to change things. 

The entry point for the overall app is actually `./index.html` as this gets loaded in by Vite. This file also contains the overall HTML base structure. 
In this file, the entrypoint to the application is loaded by importing `src/main.ts` as a script. 

This is the first point of the application where code runs.
The current code is quite simple, and just's loads in the Svelte component in the `src/App.svelte` file and puts this component into the DOM.

The entirety of the remaining application is then contained within the Svelte components.

`App.svelte` loads in the `src/components/Layout.svelte` component for overall page structure (header, ...). Then all the key data is loaded. For that a JSON object is parsed which is supposed to contain all the memory information. During development example data from `data` is loaded in. At export, it will be just a placeholder string, which needs to be filled in by the C++ application.

Global CSS classes are loaded in `src/components/Layout.svelte`, to define own global CSS `src/app.pcss` can be used. But it is advised to keep CSS contained within the Svelte components. 

`src/components` contains all the existing components for the app, for more information about the individual components, just take a look at them and the overall file structure.

`src/lib` contains pure TypeScript files. 
`types.ts` contains the type definitions for the loaded memory data, and also contains utility classes for storing and parsing that data while providing the UI with the necessary aggregated information.  
`stores.ts` contains Svelte stores. These are objects which allow storing state in Svelte which are available and changeable in the entire application.
