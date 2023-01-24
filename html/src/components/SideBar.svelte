<script lang="ts">
  import { pageState } from '../lib/stores';
  import { currentMemoryRegion } from '../lib/stores.js';
  import { ListBox, ListBoxItem } from '@skeletonlabs/skeleton';
</script>

<div class="sidebar flex flex-col h-full overflow-y-hidden">
  <div class="overflow-y-auto">
    <p class="my-1">Select the data structure you want to view:</p>

    <ListBox selected={currentMemoryRegion} class="my-1 border-gray-600/20 border-2 rounded-3xl">
      {#each $pageState.availableMemoryRegions as region, index}
        <ListBoxItem value={region} class="mb-2">{region.name}</ListBoxItem>
      {/each}
    </ListBox>
  </div>

  <div class="flex-1" />
  <div class="mt-auto my-1 py-2">
    Optionally, set a width of the Memory Region, if you have a specific 2D layout you want to visualize.
  </div>
  <input type="number" bind:value={$pageState.customMemoryWidth} />
</div>

<style>
</style>
