<script lang="ts">
  import { MemoryRegionManager } from '../lib/types';
  import MemoryRegionBlock from './MemoryRegionBlock.svelte';
  import { pageState } from '../lib/stores';

  export let MemoryRegion: MemoryRegionManager;

  // Create the object which is looped over to display all elements
  let addressObject: Iterable<number | [number, number]> | { length: number } = [];

  MemoryRegion.displaySettings.using1DSparseRepresentation = false;

  let highestIndexWidth = MemoryRegion.getHighestIndex().toString().length + 1.5 + 'ch';
  let highestValueWidth = MemoryRegion.highestTotalCount.toString().length + 1.5 + 'ch';

  $: {
    if ($pageState.showGrid == false || $pageState.customMemoryWidth == 0) {
      addressObject = MemoryRegion.getSparse1DMemoryLocations();
    } else {
      addressObject = { length: MemoryRegion.getHighestIndex() + 1 };
    }
  }
</script>

<div class="text-center m-3 w-full">
  <div class="badge">
    {MemoryRegion.name}
  </div>
  <div
    class="w-full"
    class:screen-grid={$pageState.showGrid && $pageState.customMemoryWidth <= 0}
    class:custom-grid={$pageState.showGrid && $pageState.customMemoryWidth > 0}
    style={'--cell-index-width: ' +
      highestIndexWidth +
      '; --cell-content-width: ' +
      highestValueWidth +
      '; --cell-inbetween-width: ' +
      ($pageState.showIndex
        ? MemoryRegion.getHighestIndex().toString().length + MemoryRegion.highestTotalCount.toString().length + 2.2
        : MemoryRegion.highestTotalCount.toString().length + 1.15) +
      'ch' +
      '; --cell-background-color: ' +
      ($pageState.backGroundContrastBlack ? 'black' : 'white') +
      '; --cell-text-color: ' +
      ($pageState.backGroundContrastBlack ? 'white' : 'black') +
      '; --grid-width: ' +
      $pageState.customMemoryWidth +
      ';'}
  >
    {#each addressObject as address, index}
      <!-- Check if the address is a valid object, if not we are not using a sparse array but show every element instead, so we need to use the index -->
      {#if address === undefined}
        <MemoryRegionBlock {MemoryRegion} {index} />
      {:else}
        <!-- Now check if we have a number or an array, if we have a number, this is our index, if it is an array, we actually have items left out and just symbolize that with three dots -->
        {#if typeof address == 'number'}
          <MemoryRegionBlock {MemoryRegion} index={address} />
        {:else}
          <div class="text-center text-xl inbetween-step my-auto">...</div>
        {/if}
      {/if}
    {/each}
  </div>
</div>

<style>
  :root {
    --minimum-grid-width: 60px;
    --cell-index-width: 32px;
    --cell-content-width: 32px;
    --cell-inbetween-width: 32px;
    --cell-background-color: black;
    --cell-text-color: white;
    --grid-width: 10;
  }

  .screen-grid {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    align-content: center;
    flex-direction: row;
    gap: 4px 0px;
  }

  .custom-grid {
    display: grid;
    grid-template-columns: repeat(var(--grid-width), 1fr);
    grid-gap: 0px 0px;
  }

  .inbetween-step {
    transform: translate(0px, -5px);
    min-width: var(--cell-inbetween-width);
  }
</style>
