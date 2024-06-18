<script lang="ts">
  import init, { FloatInfo } from "../fp-vis-wasm/pkg";
  import { onMount } from "svelte";
  import Values from "./Values.svelte";

  import katex from "katex";

  let numberInput: HTMLInputElement;

  enum BitType {
    sign = "sign",
    exponent = "exponent",
    mantissa = "mantissa",
  }

  type FloatPart = {
    value: string;
    bits: boolean[];
  };

  type Value = {
    fraction: string;
    decimal: string;
  };

  type FInfo = {
    hex: string;
    value: Value;
    category: string;
    error: Value;
    parts: Record<BitType, FloatPart>;
    nearby_floats: [number, Value][]
  };

  enum FloatType {
    f64 = "f64",
    f32 = "f32",
  }

  type Info = {
    value: Value;
    floats: Record<FloatType, FInfo>;
  };

  type Constant = {
    name: string,
    value: number,
  };
  type Constants = Record<FloatType, Constant[]>;

  const BitTypes = Object.values(BitType);
  const FloatTypes = Object.values(FloatType);

  let floatInfo: FloatInfo | undefined = undefined;
  let info: Info | undefined = undefined;
  let constants: Constants | undefined = undefined;

  function updateInfo(): void {
    let newFloatInfo;
    try {
      newFloatInfo = new FloatInfo(numberInput.value);
    } catch (e) {
      alert(`Error parsing input: ${e}`);
      return;
    }
    if (floatInfo !== undefined) {
      floatInfo.free();
    }
    floatInfo = newFloatInfo;
    info = floatInfo!.get_info();
  }

  let ignoreNextHashChange = false;
  function setInput(value: string) {
    numberInput.value = value;
    ignoreNextHashChange = true;
    window.location.hash = value;
  }

  function toggleBit(floatType: FloatType, i: number): void {
    floatInfo![`toggle_bit_${floatType}`](i);
    info = floatInfo!.get_info();
    setInput(info!.floats[floatType].value.fraction);
  }

  function addToBits(floatType: FloatType, n: number): void {
    floatInfo![`add_to_bits_${floatType}`](n);
    info = floatInfo!.get_info();
    setInput(info!.floats[floatType].value.fraction);
  }

  function setFloat(floatType: FloatType, n: number): void {
    floatInfo![`set_${floatType}`](n);
    info = floatInfo!.get_info();
    setInput(info!.floats[floatType].value.fraction);
  }

  function handleKeyPress(e: KeyboardEvent): void {
    if (e.key === "Enter") {
      e.preventDefault();
      updateInfo();
      window.location.hash = numberInput.value;
    }
  }

  function* bitIter(info: FInfo) {
    let i = 0;
    for (let typ of BitTypes) {
      for (let value of info.parts[typ].bits) {
        yield {
          i,
          typ,
          value,
        };
        i++;
      }
    }
  }

  const colors = {
    sign: "lightpink",
    exponent: "lightblue",
    mantissa: "lightgreen",
  };

  function getFormula(info: FInfo): string | undefined {
    const sign = info.parts.sign.value;
    const exponent = info.parts.exponent.value;
    const mantissa = info.parts.mantissa.value;

    if (info.parts.exponent.bits.every((b) => b)) {
      return undefined;
    }

    return katex.renderToString(
      `\\colorbox{${colors.sign}}{${sign}} \\times 2^{\\colorbox{${colors.exponent}}{${exponent}}} \\times \\colorbox{${colors.mantissa}}{${mantissa}}`
    );
  }

  function getHash(): string {
    return decodeURIComponent(window.location.hash.substring(1));
  }

  function onHashChange(): void {
    if (ignoreNextHashChange) {
      ignoreNextHashChange = false;
      return;
    }
    setInput(getHash());
    updateInfo();
  }

  onMount(async () => {
    numberInput.value = getHash() || "1 / 3";
    await init();
    updateInfo();
    constants = floatInfo!.constants();

    window.addEventListener("hashchange", onHashChange);
  });

  $: console.log(info);
</script>

<main>
  <input type="text" bind:this={numberInput} on:keypress={handleKeyPress} />

  {#if info}
    <br />
    <Values value={info.value} />

    {#each FloatTypes as floatType}
      {@const finfo = info.floats[floatType]}
      {@const formula = getFormula(finfo)}
      <section>
        <h1>{floatType}</h1>
        {#each constants[floatType] as constant}
          <button type="button" on:click={() => setFloat(floatType, constant.value)}>{constant.name}</button>
        {/each}
        <p>{finfo.hex}</p>
        <svg width="100%" height="30">
          <line x1="50%" y1="0" x2="50%" y2="30" style="stroke: blue; stroke-width: 3;" />
          {#each finfo.nearby_floats as nb}
            {@const x = nb[0] * 0.99 * 50 + 50 + "%"}
            {@const color = nb[1].decimal === finfo.value.decimal? "black": "gray"}
            <line x1={x} y1="0" x2={x} y2="30" style="stroke: {color}; stroke-width: 3;" />
          {/each}
        </svg>
        <table>
          <thead>
            <tr>
              {#each BitTypes as typ}
                {@const bits = finfo.parts[typ].bits.length}
                <th colspan={bits}
                  >{typ}{#if bits > 1}
                    &nbsp;({bits} bits)
                  {/if}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            <tr>
              {#each bitIter(finfo) as bit}
                <td
                  class="bit"
                  style="background-color: {colors[bit.typ]}"
                  on:click={() => toggleBit(floatType, bit.i)}
                  >{bit.value ? 1 : 0}</td
                >
              {/each}
              <td>
                <button type="button" on:click={() => addToBits(floatType, 1)}
                  >+</button
                >
              </td>
              <td>
                <button type="button" on:click={() => addToBits(floatType, -1)}
                  >-</button
                >
              </td>
            </tr>
            <tr>
              {#each BitTypes as typ}
                {@const part = finfo.parts[typ]}
                <td colspan={part.bits.length}>{part.value}</td>
              {/each}
            </tr>
          </tbody>
        </table>
        <br />
        Category: {finfo.category}
        {#if formula}
          <br />
          {@html formula}
        {/if}
        <br />
        <Values value={finfo.value} />
        <br />
        <h2>Error</h2>
        <Values value={finfo.error} />
      </section>
    {/each}
  {:else}
    Error/Loading (WebAssembly is required)
  {/if}
</main>

<style>
  main {
    line-height: 2em;
  }

  section {
    border-style: solid;
    border-width: 1px;
    margin: 1em;
    padding: 1em;
  }

  h1 {
    margin-top: 0;
  }

  table {
    margin: auto;
  }

  .bit {
    width: 1em;
    cursor: pointer;
    user-select: none;
  }
</style>
