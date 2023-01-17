<script lang="ts">
  import { drawerState } from '../lib/stores';
  import { MemoryRegionManager } from '../lib/types';
  import { drawerStore } from '@skeletonlabs/skeleton';

  export let MemoryRegion: MemoryRegionManager;
  export let index: number;

  let originalMemoryAddress: string;
  let readCount = 0;
  let writeCount = 0;
  let totalCount = 0;

  $: {
    readCount = MemoryRegion.getReadAccesses(index).length;
    writeCount = MemoryRegion.getWriteAccesses(index).length;
    totalCount = readCount + writeCount;

    originalMemoryAddress = MemoryRegion.convertIndexToAddressString(index);
  }
</script>

<div class="flex bg-cyan-700 m-0 w-56">
  <div class="bg-cyan-400 p-4 flex-1">
    {index}
  </div>
  <div class="flex flex-col flex-1 bg-cyan-800">
    {#if $drawerState.showSingleAccessTable}
      <div
        class="flex-1 bg-cyan-600"
        on:click={() => {
          drawerStore.open({
            position: 'bottom'
          });
          $drawerState.showSingleAccessTable = true;
          $drawerState.currentMemoryRegion = MemoryRegion;
          $drawerState.currentMemoryRegionIndex = index;
        }}
      >
        {totalCount}
      </div>
    {:else}
      <div
        class="flex-1 bg-cyan-600"
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
        class="flex-1 bg-orange-700"
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
