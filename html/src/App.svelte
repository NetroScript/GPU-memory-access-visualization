<Layout>

  <div class="container h-full mx-auto flex justify-center items-center">
    <div class="space-y-0.5 text-center">

      {#each addresses as address}
        <div class="flex bg-cyan-700 m-0 w-56">
          <div class="bg-cyan-400 p-4 flex-1">
            {"0x"+(address.toString(16)).padStart(8, '0')}
          </div>
          <div class="flex flex-col flex-1 bg-cyan-800">
            {#if MemoryAccesses.has(address)}
              <div class="flex-1 bg-cyan-600" on:click={() => {
						drawerStore.open( {
							position: "bottom",
						})
						activeAddress = address
						isWrite = 0}}>
                {MemoryAccesses.get(address)[0].length}</div>
              <div class="flex-1 bg-orange-700"  on:click={() => {
						drawerStore.open( {
							position: "bottom",
						})
						activeAddress = address
						isWrite = 1}}>
                {MemoryAccesses.get(address)[1].length}</div>
            {/if}
          </div>
        </div>

      {/each}

    </div>

    <style lang="postcss">
        figure {
            @apply flex relative flex-col;
        }
        figure svg,
        .img-bg {
            @apply w-64 h-64 md:w-80 md:h-80;
        }
        .img-bg {
            @apply bg-gradient-to-r from-primary-300 to-warning-300;
            @apply absolute z-[-1] rounded-full blur-[64px];
            animation: pulse 5s cubic-bezier(0, 0, 0, 0.5) infinite;
        }
        @keyframes pulse {
            50% {
                transform: scale(1.5);
            }
        }
    </style>
  </div>
  <Drawer>
    {#if activeAddress > -1}

      {#each MemoryAccesses.get(activeAddress)[isWrite] as access}
        <div class="flex bg-cyan-800 mb-1 break-words overflow-hidden">
          <div class="bg-cyan-900 p-4 flex-1 break-words max-w-full ">
            {JSON.stringify(access)}
          </div>
        </div>
      {/each}
    {/if}
  </Drawer>
</Layout>

<script lang="ts">
  import Layout from "./components/Layout.svelte";

  // Have map storing for each memory address the corresponding memory accesses
  // To have a range, we also need to store our minimum and maximum address

  // First load the data, here we are generating random data for showcase

  import {AccessInstance, type GenericInformation} from "./lib/types"


  const data: AccessInstance[] = [];

  import {Drawer, drawerStore} from '@skeletonlabs/skeleton';


  // Temporarily fill random data, for that first generate a generic info
  const info: GenericInformation = {
    GridDimensions: {
      x: 32,
      y: 1,
      z: 1
    },
    BlockDimensions:
      {
        x: 1,
        y: 1,
        z: 1
      },
    WarpSize: 32,
  }

  // Generate currentSize AccessInstances

  for (let i = 0; i < 1000; i++) {
    data.push(new AccessInstance(
      // As address generate random 64 bit hex value
      Math.floor(Math.random() * 2**8).toString(16),
      // Fill in random thread id which is between 0 and 31
      Math.floor(Math.random() * 32),
      // Same for the block
      Math.floor(Math.random() * 32),
      // Random either read or write
      Math.random() > 0.5,
      info
    ));
  }

  // Have the data structure storing all access to the corresponding memory location
  // This maps from the address (as a string) to a tuple of all read accesses as array, and all write accesses as array
  const MemoryAccesses = new Map<bigint, [AccessInstance[], AccessInstance[]]>();

  let lowestAddress = 2n**64n;
  let highestAddress = 0n;

  // Iterate all of our data
  data.forEach(entry => {
    if (entry.address < lowestAddress) {
      lowestAddress = entry.addressInteger;
    }
    if (entry.address > highestAddress) {
      highestAddress = entry.addressInteger;
    }

    // Check if we already have an entry for this address
    if (!MemoryAccesses.has(entry.addressInteger)) {
      // If not, create a new entry
      MemoryAccesses.set(entry.addressInteger, [[], []]);
    }

    // Get the entry
    const [read, write] = MemoryAccesses.get(entry.addressInteger)!;

    // Check if we have a read or write
    if (entry.isRead) {
      read.push(entry);
    } else {
      write.push(entry);
    }
  });

  // Make a string array of all addresses within bounds
  const addresses : bigint[] = [];
  for (let i = lowestAddress; i <= highestAddress; i++) {
    addresses.push(i);
  }

  let activeAddress = -1n;
  let isWrite = -1;

  BigInt.prototype.toJSON = function() { return this.toString() }
</script>
