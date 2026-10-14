// The web Host's way into the DOM, which emscripten's HTML5 API does not wrap.
// Linked into modeler.js with `--js-library` (see build.rs).

addToLibrary({
  // File transfer. Bytes move between the browser and emscripten's in-memory
  // filesystem, where the Rust side reads and writes ordinary paths.
  $fileTransfer: {
    // Paths of picked files written into the filesystem, oldest first.
    picked: [],
    // The open picker's input, held so it is not collected before `change`.
    input: null,
  },

  duck_pick_file__deps: ["$fileTransfer", "$FS", "$UTF8ToString"],
  duck_pick_file: (accept) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = UTF8ToString(accept);
    input.addEventListener("change", async () => {
      fileTransfer.input = null;
      const file = input.files[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      FS.mkdirTree("/tmp/import");
      const path = `/tmp/import/${file.name}`;
      FS.writeFile(path, bytes);
      fileTransfer.picked.push(path);
    });
    fileTransfer.input = input;
    input.click();
  },

  duck_take_picked_file__deps: ["$fileTransfer", "$stringToUTF8"],
  duck_take_picked_file: (buf, cap) => {
    const path = fileTransfer.picked.shift();
    return path === undefined ? 0 : stringToUTF8(path, buf, cap);
  },

  duck_download_file__deps: ["$FS", "$UTF8ToString"],
  duck_download_file: (pathPtr) => {
    const path = UTF8ToString(pathPtr);
    const bytes = FS.readFile(path);
    FS.unlink(path);
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = path.split("/").pop();
    link.click();
    // The click has started the download by the next task.
    setTimeout(() => URL.revokeObjectURL(url));
  },
});
