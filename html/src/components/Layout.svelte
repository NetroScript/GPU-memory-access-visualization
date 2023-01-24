<script>
  import '../theme.pcss';
  import '@skeletonlabs/skeleton/styles/all.css';
  import '../app.pcss';
  import { AppShell, AppBar, LightSwitch, tooltip, Modal, modalStore } from '@skeletonlabs/skeleton';
  import Icon from '@iconify/svelte';
  import baselineLightbulb from '@iconify/icons-ic/baseline-lightbulb';
  import outlineSplitscreen from '@iconify/icons-ic/outline-splitscreen';
  import outlineRectangle from '@iconify/icons-ic/outline-rectangle';
  import baselineMenu from '@iconify/icons-ic/baseline-menu';
  import baselineFormatListNumbered from '@iconify/icons-ic/baseline-format-list-numbered';
  import baselineLinearScale from '@iconify/icons-ic/baseline-linear-scale';
  import roundGridView from '@iconify/icons-ic/round-grid-view';
  import { pageState, drawerState } from '../lib/stores';
  import SideBar from './SideBar.svelte';

  const openInfoModal = () => {
    modalStore.trigger({
      type: 'alert',
      buttonTextCancel: 'Close',
      title: 'Information',
      body: '<div class="overflow-y-auto max-h-[200px]">This is a Utility which allows you to visualize memory accesses on the GPU. <br> Using a C++ Header only library you can export the necessary data. <br/> This is then visualized in the center screen. <br><br>To the left side you can select the memory structure you want to visualize currently. On the bottom left, you can set a memory width, which will cause the cells to automatically align to that width, and break after this width. Y goes down. <br><br>On this top bar, you also have a few utility buttons to change what is visualized. They have tooltips on hover, but the best thing would be to just try out the different buttons. <br><br> This application is a single page app made with Svelte and skeleton.dev as UI framework.</div>'
    });
  };
</script>

<!-- App Shell -->
<AppShell slotSidebarLeft="bg-surface-900/20 dark:bg-surface-900/70 w-80 p-4" slotPageContent="overflow-x-auto">
  <svelte:fragment slot="header">
    <!-- App Bar -->
    <AppBar>
      <svelte:fragment slot="lead">
        <strong class="text-xl uppercase">Memory Visualization</strong>
        <strong class="px-3">v1.0.0</strong>
      </svelte:fragment>
      <svelte:fragment slot="trail">
        <button
          class="btn-icon variant-soft min-w-[50px]"
          on:click={() => ($pageState.showGrid = !$pageState.showGrid)}
          use:tooltip={{
            content: 'Toggle between showing the memory in a grid and one long line',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl min-w-[32px]"
            height="32"
            icon={$pageState.showGrid ? baselineLinearScale : roundGridView}
            width="32"
          />
        </button>
        <button
          class="btn-icon variant-soft min-w-[50px]"
          on:click={() => ($pageState.showIndex = !$pageState.showIndex)}
          use:tooltip={{
            content: 'Toggle between showing the index of the memory',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl min-w-[32px]"
            height="32"
            icon={$pageState.showIndex ? baselineMenu : baselineFormatListNumbered}
            width="32"
          />
        </button>
        <button
          class="btn-icon variant-soft min-w-[50px]"
          on:click={() => ($drawerState.showSingleAccessTable = !$drawerState.showSingleAccessTable)}
          use:tooltip={{
            content: 'Toggle between showing read and write separately and showing the total access count',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl min-w-[32px]"
            height="32"
            icon={$drawerState.showSingleAccessTable ? outlineSplitscreen : outlineRectangle}
            width="32"
          />
        </button>
        <button
          class="btn-icon variant-soft min-w-[50px]"
          on:click={() => ($pageState.backGroundContrastBlack = !$pageState.backGroundContrastBlack)}
          use:tooltip={{ content: 'Toggle between light and dark background for the memory grid', position: 'bottom' }}
        >
          <Icon
            class="drop-shadow-xl min-w-[32px]"
            color={!$pageState.backGroundContrastBlack ? 'black' : 'white'}
            height="32"
            icon={baselineLightbulb}
            width="32"
          />
        </button>
        <LightSwitch />
        <button class="btn variant-ghost btn-sm" on:click={openInfoModal}>About</button>
      </svelte:fragment>
    </AppBar>
  </svelte:fragment>
  <svelte:fragment slot="sidebarLeft">
    <SideBar />
  </svelte:fragment>
  <Modal />
  <slot />
</AppShell>
