<div class="space-y-0.5 text-center flex flex-col m-3">

  <div class="badge">
    {MemoryRegion.name}
  </div>
  {#each addressObject as address, index}

    <!-- Check if the address is a valid object, if not we are not using a sparse array but show every element instead, so we need to use the index -->
    {#if address == undefined}
      <MemoryRegionBlock MemoryRegion="{MemoryRegion}" index="{index}" />
    {:else}
      <!-- Now check if we have a number or an array, if we have a number, this is our index, if it is an array, we actually have items left out and just symbolize that with three dots -->
      {#if typeof address == "number"}
        <MemoryRegionBlock MemoryRegion="{MemoryRegion}" index="{address}" />
      {:else}
        <div>
          ...
        </div>
      {/if}
    {/if}

  {/each}

</div>

<script lang="ts">
  import { MemoryRegionManager } from "../lib/types";
  import MemoryRegionBlock from "./MemoryRegionBlock.svelte";

  export let MemoryRegion: MemoryRegionManager;

  // Create the object which is looped over to display all elements
  let addressObject: Iterable<number | [number, number]> | { length: number } = [];

  MemoryRegion.displaySettings.using1DSparseRepresentation = false;

  $: {
    if (MemoryRegion.displaySettings.using1DSparseRepresentation) {
      addressObject = MemoryRegion.getSparse1DMemoryLocations();
    } else {
      addressObject = { length: MemoryRegion.getHighestIndex()+1 };
    }
  }
</script>
