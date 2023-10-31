<script lang="ts">
  type Value = {
    fraction: string;
    decimal: string;
  };

  export let value: Value;
</script>

<div class="values">
  = {@html value.decimal}
  {#if value.fraction.includes("/")}
    <br />
    = {value.fraction}
  {/if}
</div>

<style>
  .values {
    display: inline-block;
    text-align: left;
  }
</style>
