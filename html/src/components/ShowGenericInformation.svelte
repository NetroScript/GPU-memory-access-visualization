<script lang="ts">
  import Icon from '@iconify/svelte';
  import roundWarning from '@iconify/icons-ic/round-warning';

  import { currentMemoryRegion } from '../lib/stores';
  import { AccordionGroup, AccordionItem } from '@skeletonlabs/skeleton';
</script>

{#if $currentMemoryRegion.name !== undefined}
  <div class="variant-glass p-2 rounded-3xl text-sm mt-4">
    <AccordionGroup spacing="space-y-1" collapse={false}>
      <AccordionItem>
        <svelte:fragment slot="summary"
          ><div class="text-center font-bold">Active Kernel Information</div></svelte:fragment
        >
        <svelte:fragment slot="content">
          <div class="text-center font-mono">Grid Sizing</div>
          <div class="flex flex-row justify-around">
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">X</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.GridDimensions.x}
              </div>
            </div>
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Y</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.GridDimensions.y}
              </div>
            </div>
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Z</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.GridDimensions.z}
              </div>
            </div>
          </div>

          <div class="text-center font-mono mt-3">Block Sizing</div>
          <div class="flex flex-row justify-around">
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">X</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.BlockDimensions.x}
              </div>
            </div>
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Y</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.BlockDimensions.y}
              </div>
            </div>
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Z</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.BlockDimensions.z}
              </div>
            </div>
          </div>

          <div class="text-center font-mono mt-3 mb-2">Warp Size</div>
          <div class="flex flex-row justify-around">
            <div class="flex flex-col justify-center">
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.kernelSettings.WarpSize}
              </div>
            </div>
          </div>
        </svelte:fragment>
      </AccordionItem>
      <AccordionItem>
        <svelte:fragment slot="summary"
          ><div class="text-center font-bold">Active Memory Information</div></svelte:fragment
        >
        <svelte:fragment slot="content">
          <div class="text-center font-mono mb-2">Name</div>
          <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-100">
            {$currentMemoryRegion.name}
          </div>

          <div class="text-center font-mono mt-3 mb-2">Address Range</div>
          <div
            class="flex flex-row justify-around cursor-help"
            title="The memory region contains a total of {$currentMemoryRegion.numberOfElements} elements at a size of {$currentMemoryRegion.sizeOfSingleElement} bytes per element."
          >
            <div class="flex flex-row justify-center">
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.startAddress}
              </div>
              <div class="text-center py-1 px-3  min-w-[16px]">-</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[64px]">
                {$currentMemoryRegion.endAddress}
              </div>
            </div>
          </div>

          <div class="text-center font-mono mt-3 mb-2">Logged Accesses</div>
          <div class="flex flex-row justify-around">
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Reading</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[92px]">
                {$currentMemoryRegion.totalReads}
              </div>
            </div>
            <div class="flex flex-col justify-center">
              <div class="text-center text-xs font-mono opacity-60">Writing</div>
              <div class="text-center py-1 px-3 variant-glass rounded-xl min-w-[92px]">
                {$currentMemoryRegion.totalWrites}
              </div>
            </div>
          </div>
        </svelte:fragment>
      </AccordionItem>
    </AccordionGroup>

    {#if $currentMemoryRegion.kernelSettings.CurrentSize > $currentMemoryRegion.kernelSettings.OriginalSize}
      <aside
        class="alert variant-filled-error mt-4 cursor-help"
        title="{$currentMemoryRegion.kernelSettings.OriginalSize} elements were allocated, {$currentMemoryRegion
          .kernelSettings.CurrentSize} elements were accessed"
      >
        <!-- Message -->
        <div class="alert-message">
          <div class=" text-sm flex ">
            <div class="local-icon-style"><Icon icon={roundWarning} width="32" height="32" /></div>
            The memory storage which was used to store the data was too small to fit all accesses. ({$currentMemoryRegion
              .kernelSettings.CurrentSize - $currentMemoryRegion.kernelSettings.OriginalSize} more elements needed to be
            allocated when initializing)
          </div>
        </div>
      </aside>
    {/if}
  </div>
{/if}

<style>
  .local-icon-style {
    min-width: 24px;
    transform: translateX(-12px);
  }
</style>
