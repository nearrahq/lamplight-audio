// Static narration files. The only code: byte ranges, which iPhone Safari needs for <audio> and the static
// asset server doesn't do. A range request streams just the asked-for bytes out of the full file.
export default {
  async fetch(req, env) {
    const range = req.headers.get("Range");
    const res = await env.ASSETS.fetch(req.url, { headers: { "Accept-Encoding": "identity" } });
    if (!range || res.status !== 200 || res.status === 206) return res;
    let size = Number(res.headers.get("Content-Length"));
    let body = res.body;
    if (!size) { const buf = await res.arrayBuffer(); size = buf.byteLength; body = new Response(buf).body; }
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (!m || !size) return new Response(body, res);
    let start = m[1] === "" ? Math.max(0, size - Number(m[2])) : Number(m[1]);
    let end = m[1] === "" || m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
    if (start > end || start >= size) {
      return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}`, "Access-Control-Allow-Origin": "*" } });
    }
    const len = end - start + 1;
    let seen = 0;
    const cut = new TransformStream({
      transform(chunk, ctl) {
        const a = Math.max(start - seen, 0), b = Math.min(end + 1 - seen, chunk.byteLength);
        if (b > a) ctl.enqueue(chunk.subarray(a, b));
        seen += chunk.byteLength;
        if (seen > end) ctl.terminate();
      },
    });
    const h = new Headers(res.headers);
    h.set("Content-Range", `bytes ${start}-${end}/${size}`);
    h.set("Content-Length", String(len));
    h.set("Accept-Ranges", "bytes");
    body.pipeTo(cut.writable).catch(() => {});
    return new Response(cut.readable, { status: 206, headers: h });
  },
};
