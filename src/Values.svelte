<script lang="ts">
type Value = {
	fraction: string;
	decimal: string;
	hex_literal: string;
};

let { value }: { value: Value } = $props();
</script>

<div class="values">
	{#if value.hex_literal}
		=&nbsp;{value.hex_literal}
		<br />
	{/if}
	=&nbsp;{@html value.decimal}
	{#if value.fraction.includes("/")}
		<br />
		=&nbsp;{value.fraction}
	{/if}
</div>

<style>
	.values {
		display: inline-block;
		text-align: left;
		overflow-wrap: anywhere;
		margin-left: 100px;
		margin-right: 100px;
	}
</style>
