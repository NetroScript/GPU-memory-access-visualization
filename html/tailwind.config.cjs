/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: 'class',
	content: ['./src/**/*.{html,js,svelte,ts}', require('path').join(require.resolve('@skeletonlabs/skeleton'), '../**/*.{html,js,svelte,ts}')],
	theme: {
		extend: {
			colors: {
				access: {
					read: 'rgb(var(--read-accesses) / <alpha-value>)',
					write: 'rgb(var(--write-accesses) / <alpha-value>)',
					all: 'rgb(var(--all-accesses) / <alpha-value>)',
				}
			}
		},
	},
	plugins: [require('@tailwindcss/forms'),require('@tailwindcss/typography'),require('@tailwindcss/line-clamp'),...require('@skeletonlabs/skeleton/tailwind/skeleton.cjs')()]
}
