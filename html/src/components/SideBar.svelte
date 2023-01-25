<script lang="ts">
  import { pageState } from '../lib/stores';
  import { currentMemoryRegion, ListPlaceHolder } from '../lib/stores';
  import { ListBox, ListBoxItem } from '@skeletonlabs/skeleton';
  import ShowGenericInformation from './ShowGenericInformation.svelte';
</script>

<div class="sidebar flex flex-col h-full overflow-y-hidden">
  <div class="overflow-y-auto">
    <p class="my-1">Select the data structure you want to view:</p>

    <ListBox selected={currentMemoryRegion} class="my-1 border-gray-600/20 border-2 rounded-3xl">
      <ListBoxItem value={ListPlaceHolder} class="mb-2">None</ListBoxItem>
      {#each $pageState.availableMemoryRegions as region, index}
        <ListBoxItem value={region} class="mb-2">{region.name}</ListBoxItem>
      {/each}
    </ListBox>
  </div>

  <div class="flex-1" />
  <ShowGenericInformation />
  <div class="mt-auto my-1 py-2">
    Optionally, set a width of the Memory Region, if you have a specific 2D layout you want to visualize.
  </div>
  <input type="number" bind:value={$pageState.customMemoryWidth} />
</div>

<style>
</style>
