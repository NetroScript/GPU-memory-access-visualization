<script lang="ts">
  import { cubeHelixLookup, drawerState } from '../lib/stores';
  import { MemoryRegionManager } from '../lib/types';
  import { drawerStore, tooltip } from '@skeletonlabs/skeleton';
  import { pageState } from '../lib/stores';

  export let MemoryRegion: MemoryRegionManager;
  export let index: number;

  let originalMemoryAddress: string;
  let readCount = 0;
  let writeCount = 0;
  let totalCount = 0;

  let readOpacity = 0;
  let writeOpacity = 0;
  let totalOpacity = 0;

  $: {
    readCount = MemoryRegion.getReadAccesses(index).length;
    writeCount = MemoryRegion.getWriteAccesses(index).length;
    totalCount = readCount + writeCount;

    // If the maximum is 0, we are dividing by 0, so we set the opacity to 0 in the case of NaN
    readOpacity = readCount / MemoryRegion.highestReadCount || 0;
    writeOpacity = writeCount / MemoryRegion.highestWriteCount || 0;
    totalOpacity = totalCount / MemoryRegion.highestTotalCount || 0;
    if ($pageState.customTotalAccessCount > 0) {
      totalOpacity = totalCount / $pageState.customTotalAccessCount;
    }

    originalMemoryAddress = MemoryRegion.convertIndexToAddressString(index);
  }
</script>

<div class="flex m-0 border-gray-600 border-[1px] cell-container">
  {#if $pageState.showIndex}
    <div class="py-1 flex-1 m-auto high-contrast-stroke font-black text-white index-cell">
      {index}
    </div>
  {/if}
  <div class="flex flex-col flex-1" title={'Index: ' + index}>
    {#if $pageState.showCombinedAccess}
      <div
        class="flex-1 flex justify-center bg-access-all  font-black high-contrast-stroke items-center high-contrast-text-shadow content-cell"
        style="--tw-bg-opacity: {totalOpacity};{$pageState.useCustomColorScheme
          ? 'background-color: ' + $cubeHelixLookup(totalOpacity) + ';'
          : ''}"
        on:click={() => {
          drawerStore.open({
            position: 'bottom'
          });
          $drawerState.showSingleAccessTable = true;
          $drawerState.currentMemoryRegion = MemoryRegion;
          $drawerState.currentMemoryRegionIndex = index;
        }}
      >
        <div>{totalCount}</div>
      </div>
    {:else}
      <div
        class="flex-1 bg-access-read high-contrast-text-shadow content-cell"
        style="--tw-bg-opacity: {readOpacity}"
        on:click={() => {
          drawerStore.open({
            position: 'bottom'
          });
          $drawerState.showSingleAccessTable = false;
          $drawerState.currentMemoryRegion = MemoryRegion;
          $drawerState.currentMemoryRegionIndex = index;
          $drawerState.showReadAccess = true;
        }}
      >
        {readCount}
      </div>
      <div
        class="flex-1 bg-access-write high-contrast-text-shadow content-cell"
        style="--tw-bg-opacity: {writeOpacity}"
        on:click={() => {
          drawerStore.open({
            position: 'bottom'
          });
          $drawerState.showSingleAccessTable = false;
          $drawerState.currentMemoryRegion = MemoryRegion;
          $drawerState.currentMemoryRegionIndex = index;
          $drawerState.showReadAccess = false;
        }}
      >
        {writeCount}
      </div>
    {/if}
  </div>
</div>

<style>
  .index-cell {
    border-right: 2px solid gray;
    min-width: var(--cell-index-width);
  }

  .content-cell {
    min-width: var(--cell-content-width);
    color: var(--cell-text-color);
  }

  .cell-container {
    background-color: var(--cell-background-color);
  }
</style>
