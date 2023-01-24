import { derived, writable } from 'svelte/store';
import type { MemoryRegionManager } from './types';

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
  availableMemoryRegions: MemoryRegionManager[];
  showGrid: boolean;
  customMemoryWidth: number;
}

export const ListPlaceHolder = {
  // eslint-disable-next-line no-prototype-builtins,@typescript-eslint/ban-types
  includes: (object: {}) => object.hasOwnProperty('isPlaceHolder'),
  isPlaceHolder: true
};

function createDrawerStateStore() {
  const { subscribe, set, update } = writable<DrawerState>({
    currentMemoryRegion: null,
    currentMemoryRegionIndex: -1,
    showSingleAccessTable: false,
    showReadAccess: true
  });

  return {
    subscribe,
    set,
    update,
    getCurrentData: () => {
      // Return a copy of the current data
      let currentData: DrawerState;
      subscribe((value) => (currentData = value))();
      // On the current data, we want to return a copy of the current memory region information
      if (currentData.showSingleAccessTable) {
        return currentData.currentMemoryRegion.getAllAccesses(currentData.currentMemoryRegionIndex);
      }
      if (currentData.showReadAccess) {
        return currentData.currentMemoryRegion.getReadAccesses(currentData.currentMemoryRegionIndex);
      } else {
        return currentData.currentMemoryRegion.getWriteAccesses(currentData.currentMemoryRegionIndex);
      }
    }
  };
}

export const drawerState = createDrawerStateStore();
export const pageState = writable<PageState>({
  backGroundContrastBlack: true,
  showIndex: false,
  availableMemoryRegions: [],
  showGrid: true,
  customMemoryWidth: 0
});

export const currentMemoryRegion = writable<MemoryRegionManager | typeof ListPlaceHolder>(ListPlaceHolder);
