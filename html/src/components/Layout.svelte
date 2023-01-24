<script>
  import '../theme.pcss';
  import '@skeletonlabs/skeleton/styles/all.css';
  import '../app.pcss';
  import { AppShell, AppBar, Drawer, tooltip } from '@skeletonlabs/skeleton';
  import Icon from '@iconify/svelte';
  import baselineLightbulb from '@iconify/icons-ic/baseline-lightbulb';
  import outlineSplitscreen from '@iconify/icons-ic/outline-splitscreen';
  import outlineRectangle from '@iconify/icons-ic/outline-rectangle';
  import baselineMenu from '@iconify/icons-ic/baseline-menu';
  import baselineFormatListNumbered from '@iconify/icons-ic/baseline-format-list-numbered';
  import { pageState, drawerState } from '../lib/stores';
</script>

<!-- App Shell -->
<AppShell slotSidebarLeft="bg-surface-500/5 w-56 p-4">
  <svelte:fragment slot="header">
    <!-- App Bar -->
    <AppBar>
      <svelte:fragment slot="lead">
        <strong class="text-xl uppercase">Memory Visualization</strong>
        <strong class="px-3">v1.0.0</strong>
      </svelte:fragment>
      <svelte:fragment slot="trail">
        <div
          class="cursor-pointer"
          on:click={() => ($pageState.showIndex = !$pageState.showIndex)}
          use:tooltip={{
            content: 'Toggle between showing the index of the memory',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl"
            height="32"
            icon={$pageState.showIndex ? baselineMenu : baselineFormatListNumbered}
            width="32"
          />
        </div>
        <div
          class="cursor-pointer"
          on:click={() => ($drawerState.showSingleAccessTable = !$drawerState.showSingleAccessTable)}
          use:tooltip={{
            content: 'Toggle between showing read and write separately and showing the total access count',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl"
            height="32"
            icon={$drawerState.showSingleAccessTable ? outlineSplitscreen : outlineRectangle}
            width="32"
          />
        </div>
        <div
          class="cursor-pointer"
          on:click={() => ($pageState.backGroundContrastBlack = !$pageState.backGroundContrastBlack)}
          use:tooltip={{ content: 'Toggle between light and dark background for the memory grid', position: 'bottom' }}
        >
          <Icon
            class="drop-shadow-xl"
            color={!$pageState.backGroundContrastBlack ? 'black' : 'white'}
            height="32"
            icon={baselineLightbulb}
            width="32"
          />
        </div>
        <button class="btn btn-ghost btn-sm">About</button>
      </svelte:fragment>
    </AppBar>
  </svelte:fragment>

  <slot />
</AppShell>
