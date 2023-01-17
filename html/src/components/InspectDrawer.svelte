<script lang="ts">
  import { drawerState } from '../lib/stores';
  import { Drawer } from '@skeletonlabs/skeleton';
</script>

<Drawer>
  <div class="mx-auto text-center text-2xl my-3">
    Active Inspection: {$drawerState.currentMemoryRegion.name} ({$drawerState.showSingleAccessTable
      ? 'All'
      : $drawerState.showReadAccess
      ? 'Read'
      : 'Write'})
  </div>

  {#if $drawerState.currentMemoryRegionIndex > -1}
    {#each drawerState.getCurrentData() as access}
      <div class="flex bg-cyan-800 mb-1 break-words overflow-hidden">
        <div class="bg-cyan-900 p-4 flex-1 break-words max-w-full">
          {JSON.stringify(access)}
        </div>
      </div>
    {/each}
  {/if}
</Drawer>
