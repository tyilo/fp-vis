<script lang="ts">
let span: HTMLElement;
let { value, onchange }: { value: string; onchange: (value: string) => void } =
	$props();

function onkeydown(e: KeyboardEvent) {
	if (e.key === "Enter") {
		span.blur();
		e.preventDefault();
	}
}

function onblur() {
	onchange(value);
}
</script>

<span bind:this={span} contenteditable bind:textContent={value} {onkeydown} {onblur} tabindex="0" role="textbox"></span>
