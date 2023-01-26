<script lang="ts">
  import '../app.pcss';
  import {
    AppShell,
    AppBar,
    LightSwitch,
    tooltip,
    Modal,
    modalStore,
    type ModalComponent
  } from '@skeletonlabs/skeleton';
  import { slide } from 'svelte/transition';
  import Icon from '@iconify/svelte';
  import baselineLightbulb from '@iconify/icons-ic/baseline-lightbulb';
  import outlineSplitscreen from '@iconify/icons-ic/outline-splitscreen';
  import outlineRectangle from '@iconify/icons-ic/outline-rectangle';
  import baselineMenu from '@iconify/icons-ic/baseline-menu';
  import baselineFormatListNumbered from '@iconify/icons-ic/baseline-format-list-numbered';
  import baselineLinearScale from '@iconify/icons-ic/baseline-linear-scale';
  import roundGridView from '@iconify/icons-ic/round-grid-view';
  import twotoneColorLens from '@iconify/icons-ic/twotone-color-lens';
  import { pageState, drawerState } from '../lib/stores';
  import SideBar from './SideBar.svelte';
  import ModalColorSelection from './ModalColorSelection.svelte';

  const openInfoModal = () => {
    modalStore.trigger({
      type: 'alert',
      buttonTextCancel: 'Close',
      title: 'Information',
      body: '<div class="overflow-y-auto">This is a Utility which allows you to visualize memory accesses on the GPU. Using a C++ Header only library you can export the necessary data. This is then visualized in the center screen. <br> You achieve the fastest loading times by dragging and dropping a <code>.json</code> into the empty window, as the parsing of just a file is much faster than using the HTML DOM parser on page load. <br><br>To the left side you can select the memory structure you want to visualize currently. On the bottom left, you can set a memory width, which will cause the cells to automatically align to that width, and break after this width. Y goes down. <br> When showing both read and write accesses, the upper part <div class="text-access-read inline">(blue)</div> are read accesses, the lower part <div class="text-access-write inline">(orange)</div> are write accesses. When toggling to total accesses, the <div class="text-access-all inline high-contrast-text-shadow-white">purple</div> block represents the combined read and write accesses.<br> When showing the index, it is visible to the left of the read/write counts. <br><br>On this top bar, you also have a few utility buttons to change what is visualized. They have tooltips on hover, but the best thing would be to just try out the different buttons. <br><br> This application is a single page app made with Svelte and skeleton.dev as UI framework.</div>'
    });
  };

  const openColorModal = () => {
    const component: ModalComponent = {
      ref: ModalColorSelection
    };

    modalStore.trigger({
      type: 'component',
      component
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
        {#if $pageState.showCombinedAccess}
          <button
            class="btn-icon variant-soft min-w-[50px "
            on:click={openColorModal}
            use:tooltip={{
              content: 'Open a menu to configure the color scale of the combined accesses',
              position: 'bottom'
            }}
            transition:slide
          >
            <Icon
              class="drop-shadow-xl min-w-[32px] dark:text-primary-400 text-primary-600"
              height="32"
              icon={twotoneColorLens}
              width="32"
            />
          </button>
          <div class="pl-1 mx-3 bg-surface-400/30 min-h-[30px]" />
        {/if}
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
          on:click={() => ($pageState.showCombinedAccess = !$pageState.showCombinedAccess)}
          use:tooltip={{
            content: 'Toggle between showing read and write separately and showing the total access count',
            position: 'bottom'
          }}
        >
          <Icon
            class="drop-shadow-xl min-w-[32px]"
            height="32"
            icon={$pageState.showCombinedAccess ? outlineSplitscreen : outlineRectangle}
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
  <Modal width="w-full max-w-[900px]" regionBody="max-h-[600px] overflow-hidden" />
  <slot />
</AppShell>
