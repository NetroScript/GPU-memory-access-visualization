<script lang="ts">
  import Layout from './components/Layout.svelte';

  // Have map storing for each memory address the corresponding memory accesses
  // To have a range, we also need to store our minimum and maximum address

  // First load the data, here we are generating random data for showcase

  BigInt.prototype.toJSON = function () {
    return this.toString();
  };

  import { AccessInstance, type GenericInformation, MemoryRegionManager, type OutputJSON } from './lib/types';

  import { Drawer, drawerStore } from '@skeletonlabs/skeleton';
  import { dummyData } from './lib/loadDummyData';
  import InspectDrawer from './components/InspectDrawer.svelte';
  import VisualizeMemoryRegion from './components/VisualizeMemoryRegion.svelte';

  // Load the general data
  // We actually do a JSON.parse here, as it is faster using the JSON parser than for the browser to evaluate an existing JSON object in JavaScript
  // This is because the parser for the DOM needs to do more complex work.
  // Additionally this leaves us an easy entry point which will not get simplified away (which normally a comment might be)

  // But actually for development mode load in a file
  const MemoryData: OutputJSON = JSON.parse(import.meta.env.DEV ? dummyData : `//JS_TEMPLATE`);

  const info = MemoryData.GlobalSettings;

  // Generate AccessInstances from the pure data
  const data: AccessInstance[] = MemoryData.MemoryAccessLogs.map(
    ([address, blockID, threadID, isRead]) => new AccessInstance(address, blockID, threadID, isRead, info)
  );

  // Generate MemoryRegionManagers
  const memoryRegions: MemoryRegionManager[] = MemoryData.MemoryRegions.map(
    (region) => new MemoryRegionManager(region)
  );

  // Store all our memory accesses in the memoryRegions
  // This is relatively slow as we need to iterate over all memory accesses and then again over all memory regions, maybe this can be optimized, but the number of memory Regions should be low, so this should not be a problem
  data.forEach((access) => {
    memoryRegions.forEach((region) => {
      region.addMemoryAccessIfInRegion(access);
    });
  });

  // For each memory region, bake all data which gets processed after the data is loaded
  memoryRegions.forEach((region) => {
    region.bake();
  });
</script>

<Layout>
  <div class="container h-full mx-auto flex justify-center items-center flex flex-row">
    {#each memoryRegions as memoryRegion}
      <VisualizeMemoryRegion MemoryRegion={memoryRegion} />
    {/each}
  </div>

  <InspectDrawer />
</Layout>
