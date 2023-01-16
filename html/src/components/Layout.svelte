<script>
    import '@skeletonlabs/skeleton/themes/theme-skeleton.css';
    import '@skeletonlabs/skeleton/styles/all.css';
    import '../app.pcss';
    import {AppShell, AppBar, Drawer} from '@skeletonlabs/skeleton';

</script>

<!-- App Shell -->
<AppShell slotSidebarLeft="bg-surface-500/5 w-56 p-4">
    <svelte:fragment slot="header">
        <!-- App Bar -->
        <AppBar>
            <svelte:fragment slot="lead">
                <strong class="text-xl uppercase">Memory Visualization</strong>
            </svelte:fragment>
            <svelte:fragment slot="trail">
                <strong>v1.0.0</strong>
            </svelte:fragment>
        </AppBar>
    </svelte:fragment>

    <slot />
</AppShell>

<style global>


</style>
