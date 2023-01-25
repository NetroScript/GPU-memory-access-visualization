<script lang="ts">
  import { AccessInstance } from '../lib/types';

  export let access: AccessInstance;
  export let index = 0;
</script>

<div class="max-w-ful">
  <div class="flex flex-row items-center variant-glass rounded-xl p-3 mx-3 my-2 flex-wrap">
    <div class="self-center justify-self-center text-3xl variant-glass-primary py-2 px-5 rounded-[100px] mx-auto">
      {index}
    </div>
    <div class="flex flex-1 flex-row justify-around items-center flex-wrap">
      <div class="section-container">
        <div class="text-center font-mono">Thread ID:</div>
        <div class="flex flex-row justify-around items-center px-2 gap-2">
          <div class="description-container">
            <div class="description-prefix">X:</div>
            <div class="description-content">
              {access.blockID.x}
            </div>
          </div>
          <div class="description-container">
            <div class="description-prefix">Y:</div>
            <div class="description-content">
              {access.blockID.y}
            </div>
          </div>
          <div class="description-container">
            <div class="description-prefix">Z:</div>
            <div class="description-content">
              {access.blockID.z}
            </div>
          </div>
          <div class="description-container pl-2">
            <div class="description-prefix">Global:</div>
            <div class="description-content">
              {access.threadIdGlobal}
            </div>
          </div>
        </div>
      </div>
      <div class="section-container">
        <div class="text-center font-mono">Block ID:</div>
        <div class="flex flex-row justify-around items-center px-2 gap-2">
          <div class="description-container">
            <div class="description-prefix">X:</div>
            <div class="description-content">
              {access.blockID.x}
            </div>
          </div>
          <div class="description-container">
            <div class="description-prefix">Y:</div>
            <div class="description-content">
              {access.blockID.y}
            </div>
          </div>
          <div class="description-container">
            <div class="description-prefix">Z:</div>
            <div class="description-content">
              {access.blockID.z}
            </div>
          </div>
          <div class="description-container pl-2">
            <div class="description-prefix">Global:</div>
            <div class="description-content">
              {access.blockIdGlobal}
            </div>
          </div>
        </div>
      </div>
    </div>
    <div
      class="py-3 my-1 px-4 text-xl rounded-3xl uppercase self-center justify-self-center mx-auto"
      class:variant-glass-success={access.isRead}
      class:variant-glass-warning={!access.isRead}
    >
      {access.isRead ? 'Read' : 'Write'}
    </div>
  </div>
</div>
