<script lang="ts">
  import { Modal, modalStore, RangeSlider, SlideToggle } from '@skeletonlabs/skeleton';
  import {
    pageState,
    cubeHelixParameters,
    cubeHelixLookup,
    cubeHelixMapFunction,
    currentMemoryRegion
  } from '../lib/stores';
  import { derived, writable } from 'svelte/store';
  import { cubehelix } from 'cubehelix';

  export let parent: Modal;
  const localCubeHelixParameters = writable(Object.assign({}, $cubeHelixParameters));
  const localCubeHelixLookup = derived(localCubeHelixParameters, ($localCubeHelixParameters) => {
    const colorArray = new Array(1000);

    const func = cubehelix($localCubeHelixParameters);

    // Fill the color array with the correct colors
    for (let i = 0; i < 1000; i++) {
      const color = func(i / 1000);
      colorArray[i] = `rgb(${color.r[0] * 255}, ${color.g[0] * 255}, ${color.b[0] * 255})`;
    }

    return (index: number) => {
      // Clamp index between 0 and 1
      index = Math.max(0, Math.min(1, index));
      return colorArray[Math.floor(index * (1000 - 1))];
    };
  });

  let gradientWidth: number;
  let multiplier = 1;
  let parametersUnchanged = true;

  $: {
    if ('name' in $currentMemoryRegion && $pageState.customTotalAccessCount) {
      multiplier = $currentMemoryRegion.highestTotalCount / $pageState.customTotalAccessCount;
    } else {
      multiplier = 1;
    }

    parametersUnchanged = JSON.stringify($cubeHelixParameters) !== JSON.stringify($localCubeHelixParameters);
  }
</script>

<div>
  <div class="w-full h-full flex flex-col align-middle justify-center items-center gap-2">
    <SlideToggle bind:checked={$pageState.useCustomColorScheme}
      >Use a cube helix color scheme for the combined view</SlideToggle
    >
    <div class="flex align-middle items-center justify-evenly">
      <div class="mr-4">Overwrite reference maximum number:</div>
      <div class=""><input type="number" class="w-8" bind:value={$pageState.customTotalAccessCount} /></div>
    </div>
    <div class="variant-soft rounded-3xl w-full h-full my-4 mx-2 p-3">
      <div class="font-black text-center mb-4">Cube Helix Settings</div>

      <div class="flex flex-col gap-4">
        <div class="flex">
          <div class="flex-1">
            <RangeSlider bind:value={$localCubeHelixParameters.start} min={0} max={5} step={0.05} ticked>
              <div class="flex justify-between items-center">
                <div class="move-down text-xs">0</div>
                <div class="flex-1 text-center">Start Parameter</div>
                <div class="move-down text-xs">5</div>
              </div>
            </RangeSlider>
          </div>
          <div class="flex align-middle items-center justify-center ml-6">
            <div class="mr-4">Value:</div>
            <input
              class="max-w-[64px] text-xs p-2"
              min="0"
              max="5"
              type="number"
              bind:value={$localCubeHelixParameters.start}
            />
          </div>
        </div>
        <div class="flex">
          <div class="flex-1">
            <RangeSlider bind:value={$localCubeHelixParameters.r} min={-10} max={30} step={0.1} ticked>
              <div class="flex justify-between items-center">
                <div class="move-down text-xs">-10</div>
                <div class="flex-1 text-center">Rotation Parameter</div>
                <div class="move-down text-xs">30</div>
              </div>
            </RangeSlider>
          </div>
          <div class="flex align-middle items-center justify-center ml-6">
            <div class="mr-4">Value:</div>
            <input
              class="max-w-[64px] text-xs p-2"
              min="-10"
              max="30"
              type="number"
              bind:value={$localCubeHelixParameters.r}
            />
          </div>
        </div>
        <div class="flex">
          <div class="flex-1">
            <RangeSlider bind:value={$localCubeHelixParameters.hue} min={0} max={10} step={0.05} ticked>
              <div class="flex justify-between items-center">
                <div class="move-down text-xs">0</div>
                <div class="flex-1 text-center">Hue Parameter</div>
                <div class="move-down text-xs">10</div>
              </div>
            </RangeSlider>
          </div>
          <div class="flex align-middle items-center justify-center ml-6">
            <div class="mr-4">Value:</div>
            <input
              class="max-w-[64px] text-xs p-2"
              min="0.00"
              max="10"
              type="number"
              bind:value={$localCubeHelixParameters.hue}
            />
          </div>
        </div>
        <div class="flex">
          <div class="flex-1">
            <RangeSlider bind:value={$localCubeHelixParameters.gamma} min={0.0} max={5} step={0.05} ticked>
              <div class="flex justify-between items-center">
                <div class="move-down text-xs">0</div>
                <div class="flex-1 text-center">Gamma Parameter</div>
                <div class="move-down text-xs">5</div>
              </div>
            </RangeSlider>
          </div>
          <div class="flex align-middle items-center justify-center ml-6">
            <div class="mr-4">Value:</div>
            <input
              class="max-w-[64px] text-xs p-2"
              min="0.01"
              max="5"
              type="number"
              bind:value={$localCubeHelixParameters.gamma}
            />
          </div>
        </div>
      </div>

      <div class="font-mono text-center mt-4">Preview</div>
      <div class="flex rounded-3xl overflow-hidden" bind:clientWidth={gradientWidth}>
        {#each Array.from({ length: Math.floor(gradientWidth / 2) }) as _, i}
          <div
            class="h-8 flex-1"
            style={`background-color: ${$localCubeHelixLookup((i / (gradientWidth / 2)) * multiplier)}`}
          />
        {/each}
      </div>
      <div class="font-mono text-center mt-4">Presets</div>
      <div class="flex justify-evenly items-center">
        <button
          class="btn variant-soft"
          on:click={() => {
            $localCubeHelixParameters = { start: 0, r: 0.6, hue: 3.0, gamma: 1.0 };
          }}
        >
          Heat
        </button>
        <button
          class="btn variant-soft"
          on:click={() => {
            $localCubeHelixParameters = { start: 0.75, r: -0.4, hue: 4.2, gamma: 1.55 };
          }}
        >
          Winter
        </button>
        <button
          class="btn variant-soft"
          on:click={() => {
            $localCubeHelixParameters = { start: 2.55, r: 0.1, hue: 5.9, gamma: 1.55 };
          }}
        >
          Blue
        </button>
        <button
          class="btn variant-soft"
          on:click={() => {
            $localCubeHelixParameters = { start: 5, r: 1.6, hue: 2.15, gamma: 7 };
          }}
        >
          Peak
        </button>
      </div>
    </div>
  </div>

  <div class="h-[1px] w-full bg-surface-400/30 mb-4" />
  <footer class="modal-footer {parent.regionFooter}">
    <button class="btn {parent.buttonNeutral}" on:click={parent.onClose}>Close</button>
    <button
      class="btn {parent.buttonPositive}"
      on:click={() => {
        $cubeHelixParameters = Object.assign({}, $localCubeHelixParameters);
      }}
      disabled={!parametersUnchanged}>Apply</button
    >
  </footer>
</div>

<style>
  .move-down {
    transform: translateY(0.5rem);
  }
</style>
