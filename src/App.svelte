<script lang="ts">
import katex from "katex";
import { onMount } from "svelte";
import init, { FloatInfo } from "../fp-vis-wasm/pkg";
import Values from "./Values.svelte";

let numberInput: HTMLInputElement;

const BitType = ["sign", "exponent", "mantissa"] as const;
type BitType = (typeof BitType)[number];

type FloatPart = {
	bits: boolean[];
	raw_value: string;
	value: string;
};

type Value = {
	fraction: string;
	decimal: string;
	hex_literal: string;
};

type FInfo = {
	hex: string;
	value: Value;
	category: string;
	error: Value;
	parts: Record<BitType, FloatPart>;
	nearby_floats: [number, Value][];
};

const FloatType = ["f64", "f32"] as const;
type FloatType = (typeof FloatType)[number];

type Info = {
	value: Value;
	floats: Record<FloatType, FInfo>;
};

type Bits = [number, number];

type Constant = {
	name: string;
	bits: Bits;
};
type Constants = Record<FloatType, Constant[]>;

let floatInfo: FloatInfo | undefined;
let info: Info | undefined;
let constants: Constants | undefined;

function currentFloatInfo(): FloatInfo {
	if (floatInfo !== undefined) {
		return floatInfo;
	}
	throw new Error("floatInfo is not initialized");
}

function updateInfo(): void {
	let newFloatInfo: FloatInfo | undefined;
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
	info = currentFloatInfo().get_info();
}

let ignoreNextHashChange = false;
function setInput(value: string) {
	numberInput.value = value;
	ignoreNextHashChange = true;
	window.location.hash = value;
}

function toggleBit(floatType: FloatType, i: number): void {
	currentFloatInfo().toggle_bit(floatType, i);
	const newInfo = currentFloatInfo().get_info();
	info = newInfo;
	setInput(newInfo.floats[floatType].value.fraction);
}

function addToBits(floatType: FloatType, n: number): void {
	currentFloatInfo().add_to_bits(floatType, n);
	const newInfo = currentFloatInfo().get_info();
	info = newInfo;
	setInput(newInfo.floats[floatType].value.fraction);
}

function setBits(floatType: FloatType, bits: Bits): void {
	currentFloatInfo().set_bits(floatType, bits);
	const newInfo = currentFloatInfo().get_info();
	info = newInfo;
	setInput(newInfo.floats[floatType].value.fraction);
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
	for (const typ of BitType) {
		for (const value of info.parts[typ].bits) {
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
		`\\colorbox{${colors.sign}}{${sign}} \\times 2^{\\colorbox{${colors.exponent}}{${exponent}}} \\times \\colorbox{${colors.mantissa}}{${mantissa}}`,
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
	constants = currentFloatInfo().constants();

	window.addEventListener("hashchange", onHashChange);
});

$: console.log(constants);
$: console.log(info);
</script>

<main>
	<input type="text" bind:this={numberInput} on:keypress={handleKeyPress} />

	{#if info}
		<br />
		<Values value={info.value} />

		{#each FloatType as floatType}
			{@const finfo = info.floats[floatType]}
			{@const formula = getFormula(finfo)}
			<details open>
				<summary>{floatType}</summary>
				{#if constants}
					{#each constants[floatType] as constant}
						<button type="button" on:click={() => setBits(floatType, constant.bits)}>{constant.name}</button>
					{/each}
				{/if}
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
							{#each BitType as typ}
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
							{#each BitType as typ}
								{@const part = finfo.parts[typ]}
								<td colspan={part.bits.length}>{part.raw_value}</td>
							{/each}
						</tr>
						<tr>
							{#each BitType as typ}
								{@const part = finfo.parts[typ]}
								<td colspan={part.bits.length}>↓</td>
							{/each}
						</tr>
						<tr>
							{#each BitType as typ}
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
			</details>
		{/each}
	{:else}
		Error/Loading (WebAssembly is required)
	{/if}
</main>

<style>
	main {
		line-height: 2em;
	}

	details {
		border-style: solid;
		border-width: 1px;
		margin: 1em;
		padding: 1em;
	}

	summary {
		display: block;
		font-size: 1.5em;
		cursor: pointer;
		user-select: none;
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
