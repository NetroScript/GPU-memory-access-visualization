<script lang="ts">
  import Layout from './components/Layout.svelte';

  // Have map storing for each memory address the corresponding memory accesses
  // To have a range, we also need to store our minimum and maximum address

  // First load the data, here we are generating random data for showcase

  BigInt.prototype.toJSON = function () {
    return this.toString();
  };

  import { AccessInstance, type GenericInformation, MemoryRegionManager, type OutputJSON } from './lib/types';

  import { Drawer, drawerStore, FileDropzone } from '@skeletonlabs/skeleton';
  import { dummyData } from './lib/loadDummyData';
  import InspectDrawer from './components/InspectDrawer.svelte';
  import VisualizeMemoryRegion from './components/VisualizeMemoryRegion.svelte';
  import { pageState } from './lib/stores';
  import { currentMemoryRegion } from './lib/stores';

  // Function to load a valid JSON file into the application as memory structure
  const loadJSON = (json: OutputJSON) => {
    const info = json.GlobalSettings;

    // Generate AccessInstances from the pure data
    const data: AccessInstance[] = json.MemoryAccessLogs.map(
      ([address, blockID, threadID, isRead]) => new AccessInstance(address, blockID, threadID, isRead, info)
    );

    // Generate MemoryRegionManagers
    const memoryRegions: MemoryRegionManager[] = json.MemoryRegions.map((region) => new MemoryRegionManager(region));

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

    // Add the loaded memory regions to the store so it can be selected
    $pageState.availableMemoryRegions = [...$pageState.availableMemoryRegions, ...memoryRegions];
  };

  // To support errors and loading new files into the template without an export, create all necessary variables
  // Template which will be replaced
  const loadData = '// JS_TEMPLATE';
  let validTemplate = true;
  // Holds dragged in files
  let files: FileList;

  // Check if we are not in development mode, but still no data was loaded
  if (loadData.startsWith('// JS_TEMPLAT') && !import.meta.env.DEV) {
    // Neither data, nor dev mode are enabled, so show error.
    validTemplate = false;
  } else {
    // Otherwise we might be in dev mode, or have data
    try {
      // For development mode load in a file
      // For production mode parse the existing data
      // Parsing JSON in a string is always faster than the JavaScript parser
      const MemoryData: OutputJSON = JSON.parse(import.meta.env.DEV ? dummyData : loadData);
      loadJSON(MemoryData);
    } catch (e) {
      // If we have an error, we are not in dev mode, but the data is invalid
      console.error(e);
      validTemplate = false;
    }
  }

  // This is a reactive statement if files changes, this only changes, if files were dropped
  $: if (files) {
    // Log the files we got
    console.log(files);

    // Now try to load the data instead of the static data included in the file
    for (const file of files) {
      // Check if it is a JSON file
      if (file.type === 'application/json') {
        // Read the file
        const reader = new FileReader();
        // Define the callback for when the file is read
        reader.onload = (e) => {
          // Parse the JSON
          const data: OutputJSON = JSON.parse(e.target.result as string) as OutputJSON;
          // Check if the data is valid
          if (data && data.GlobalSettings && data.MemoryRegions.length > 0 && data.MemoryAccessLogs.length > 0) {
            // Load the data
            loadJSON(data);

            // To not cause an event loading the same data twice, clear the files
            files = null;

            // Set the valid template to true
            validTemplate = true;
          }
        };
        // Start reading the file
        reader.readAsText(file);
      }
    }
  }
</script>

<Layout>
  <div class="h-full mx-auto flex flex-row">
    {#if validTemplate}
      {#if $currentMemoryRegion != null && !('isPlaceHolder' in $currentMemoryRegion)}
        <VisualizeMemoryRegion MemoryRegion={$currentMemoryRegion} />
      {:else}
        <div class="flex flex-col justify-center items-center w-full h-full">
          <h1 class="text-3xl font-bold opacity-30 ">Please select a memory region</h1>
          <div class="px-12 my-12 w-full">
            <FileDropzone bind:files title="You can upload additional .json files here." />
          </div>
        </div>
      {/if}
    {:else}
      <div class="flex flex-col justify-center items-center w-full h-full text-center">
        <h1 class="text-3xl font-bold opacity-30 ">
          Please load in valid data for this template.<br /> Without data this HTML file can't do anything.
        </h1>
        <div class="px-12 my-12 w-full">
          <FileDropzone bind:files title="You can alternatively drag in a valid .json as template here." />
        </div>
      </div>
    {/if}
  </div>

  <InspectDrawer />
</Layout>
