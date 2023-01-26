import { derived, writable } from 'svelte/store';
import type { MemoryRegionManager } from './types';
import { cubehelix } from 'cubehelix';

// Implement a store storing the state of the drawer information

export interface DrawerState {
  currentMemoryRegion: MemoryRegionManager;
  currentMemoryRegionIndex: number;
  showSingleAccessTable: boolean;
  showReadAccess: boolean;
}

export interface PageState {
  backGroundContrastBlack: boolean;
  showIndex: boolean;
  showCombinedAccess: boolean;
  availableMemoryRegions: MemoryRegionManager[];
  showGrid: boolean;
  customMemoryWidth: number;
  customTotalAccessCount: number;
  useCustomColorScheme: boolean;
}

export const ListPlaceHolder = {
  // eslint-disable-next-line no-prototype-builtins,@typescript-eslint/ban-types
  includes: (object: {}) => object.hasOwnProperty('isPlaceHolder'),
  isPlaceHolder: true
};

export const drawerState = writable<DrawerState>({
  currentMemoryRegion: null,
  currentMemoryRegionIndex: -1,
  showSingleAccessTable: false,
  showReadAccess: true
});
export const drawerContent = derived(drawerState, ($drawerState) => {
  if ($drawerState.currentMemoryRegion === null) {
    return [];
  }

  if ($drawerState.showSingleAccessTable) {
    return $drawerState.currentMemoryRegion.getAllAccesses($drawerState.currentMemoryRegionIndex);
  }
  if ($drawerState.showReadAccess) {
    return $drawerState.currentMemoryRegion.getReadAccesses($drawerState.currentMemoryRegionIndex);
  } else {
    return $drawerState.currentMemoryRegion.getWriteAccesses($drawerState.currentMemoryRegionIndex);
  }
});

export const pageState = writable<PageState>({
  backGroundContrastBlack: true,
  showIndex: false,
  availableMemoryRegions: [],
  showGrid: true,
  customMemoryWidth: 0,
  showCombinedAccess: false,
  customTotalAccessCount: 0,
  useCustomColorScheme: false
});

export const cubeHelixParameters = writable({ start: 0, r: 0.6, hue: 3.0, gamma: 1.0 });

export const cubeHelixMapFunction = derived(cubeHelixParameters, ($cubeHelixParameters) => {
  return cubehelix($cubeHelixParameters);
});

export const cubeHelixLookup = derived(cubeHelixMapFunction, ($cubeHelixMapFunction) => {
  const colorArray = new Array(1000);

  // Fill the color array with the correct colors
  for (let i = 0; i < 1000; i++) {
    const color = $cubeHelixMapFunction(i / 1000);
    colorArray[i] = `rgb(${color.r[0] * 255}, ${color.g[0] * 255}, ${color.b[0] * 255})`;
  }

  return (index: number) => {
    // Clamp index between 0 and 1
    index = Math.max(0, Math.min(1, index));
    return colorArray[Math.floor(index * (1000 - 1))];
  };
});

export const currentMemoryRegion = writable<MemoryRegionManager | typeof ListPlaceHolder>(ListPlaceHolder);
