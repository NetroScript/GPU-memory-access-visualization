<script lang="ts">
  import { drawerState, drawerContent } from '../lib/stores';
  import { Drawer } from '@skeletonlabs/skeleton';
  import MemoryAccessInfo from './MemoryAccessInfo.svelte';
  import { get } from 'svelte/store';

  const toggleCurrentlyShownType = () => {
    const currentlyShownType = get(drawerState);

    // If currently reading, and not the combined accesses change it to writing
    if (currentlyShownType.showReadAccess && !currentlyShownType.showSingleAccessTable) {
      drawerState.update((state) => {
        state.showReadAccess = false;
        return state;
      });
    }
    // If we are showing write accesses, and not the combined accesses change it to combined
    else if (!currentlyShownType.showReadAccess && !currentlyShownType.showSingleAccessTable) {
      drawerState.update((state) => {
        state.showReadAccess = true;
        state.showSingleAccessTable = true;
        return state;
      });
    }
    // If we are showing combined accesses, change it to reading
    else if (currentlyShownType.showReadAccess && currentlyShownType.showSingleAccessTable) {
      drawerState.update((state) => {
        state.showSingleAccessTable = false;
        return state;
      });
    }
  };
</script>

<Drawer height="h-[70%]" blur="backdrop-blur-lg">
  <div class="overflow-hidden h-full flex flex-col">
    <div class="mx-auto text-center text-2xl my-3 cursor-pointer" on:click={toggleCurrentlyShownType}>
      Address: {$drawerState.currentMemoryRegion.convertIndexToAddressString($drawerState.currentMemoryRegionIndex)} [{$drawerState.currentMemoryRegionIndex}]
      - {$drawerState.showSingleAccessTable ? 'All' : $drawerState.showReadAccess ? 'Read' : 'Write'} accesses in chronological
      order
      <div class="inline opacity-50 text-sm">(click to toggle shown accesses)</div>
    </div>
    <div class="flex flex-row justify-around items-center flex-wrap my-2">
      <div class="section-container flex-1 max-w-xl !variant-glass-secondary mx-3">
        <div class="text-center font-mono">Grid Sizing:</div>
        <div class="flex flex-row justify-around">
          <div class="description-container mx-3">
            <div class="description-prefix">X:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.GridDimensions.x}
            </div>
          </div>
          <div class="description-container mx-3">
            <div class="description-prefix">Y:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.GridDimensions.y}
            </div>
          </div>
          <div class="description-container mx-3">
            <div class="description-prefix">Z:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.GridDimensions.z}
            </div>
          </div>
        </div>
      </div>
      <div class="section-container flex-1 max-w-xl !variant-glass-secondary mx-3">
        <div class="text-center font-mono">Block Sizing:</div>
        <div class="flex flex-row justify-around">
          <div class="description-container mx-3">
            <div class="description-prefix">X:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.BlockDimensions.x}
            </div>
          </div>
          <div class="description-container mx-3">
            <div class="description-prefix">Y:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.BlockDimensions.y}
            </div>
          </div>
          <div class="description-container mx-3">
            <div class="description-prefix">Z:</div>
            <div class="description-content">
              {$drawerState.currentMemoryRegion.kernelSettings.BlockDimensions.z}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="min-w-full max-w-full bg-black/20 py-1" />

    <div class="overflow-y-auto flex-1">
      {#if $drawerState.currentMemoryRegionIndex > -1}
        {#each $drawerContent as access, index}
          <MemoryAccessInfo {access} {index} />
        {/each}
      {/if}
    </div>
  </div>
</Drawer>
