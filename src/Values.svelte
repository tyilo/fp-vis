<script lang="ts">
type Value = {
	fraction: string;
	decimal: string;
};

export let value: Value;
</script>

<div class="values">
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
