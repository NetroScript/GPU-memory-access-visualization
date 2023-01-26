declare module 'cubehelix' {
  interface HelixConfig {
    start: number;
    r: number;
    hue: number;
    gamma: number;
  }

  declare function cubehelix(overrides?: HelixConfig): (l: number) => { r: [number]; g: [number]; b: [number] };
  declare const defaultHelixConfig: HelixConfig;

  export { cubehelix, defaultHelixConfig };
}
