//! A round's drafted windows grouped into shared verify passes (cuda_lanes.zig's verifyFn).
const std = @import("std");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const be = lanes.backend;

/// A monotonic nanosecond reading: `TF_TAP_TIME` is a throwaway instrument and this module holds no
/// `Io`, so it reads the clock directly rather than growing a field for one experiment.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @intCast(@max(0, ts.sec) * std.time.ns_per_s + ts.nsec);
}

/// Windows a shared pass takes at most (a round's streams).
const round_streams = 32;

/// The drafted windows verified in shared passes of up to `cap` rows (both ranks' at TP=2), then drawn.
pub fn windows(self: anytype, wins: []const be.Window, out: []be.Verified, drafted: []const usize, row_ids: []const [mtp.max_rows]u32, cap: usize) !void {
    var seqs: [round_streams]*prompt.Seq = undefined;
    var ds: [round_streams]*mtp.Draft = undefined;
    var idss: [round_streams][]const u32 = undefined;
    var at: usize = 0;
    while (at < drafted.len) {
        var n: usize = 0;
        var rows: usize = 0;
        while (at + n < drafted.len and n < round_streams) : (n += 1) {
            const w = wins[drafted[at + n]];
            if (n > 0 and rows + w.rows() > cap) break;
            const l = try self.lane(w.stream);
            seqs[n] = &l.seq;
            ds[n] = &l.draft.?;
            idss[n] = row_ids[drafted[at + n]][0..w.rows()];
            rows += w.rows();
        }
        if (n == 1 or cap <= 1) {
            for (drafted[at..][0..n], seqs[0..n], ds[0..n], idss[0..n]) |wi, q, d, ids| {
                var t0: u64 = 0;
                if (std.c.getenv("TF_TAP_TIME") != null) t0 = nowNs();
                // a lone window is a pass over one stream's own rows: its tap ring takes them, and the
                // drafter finds them here (a batched pass writes no ring at all, which is why
                // cuda_lanes.zig gives a drafting round `cap = 1`)
                const o = self.opsTapped(wins[wi].stream);
                try o.verify(q, d, ids);
                if (std.c.getenv("TF_TAP_TIME") != null) {
                    try o.e.stream.*.synchronize();
                    const t1 = nowNs();
                    std.log.info("verify window: {d} rows, {d:.1} ms", .{ ids.len, @as(f64, @floatFromInt(t1 - t0)) / 1e6 });
                }
                if (self.drafter) |draft2| draft2.note(wins[wi].stream, q.fed, q.fed + @as(u64, @intCast(ids.len)));
            }
        } else if (self.ranks) |r| {
            try r.verifyMany(@TypeOf(self.tri()), self.tri(), seqs[0..n], ds[0..n], idss[0..n], self.logits.ptr);
        } else try self.ops().verifyMany(seqs[0..n], ds[0..n], idss[0..n], self.logits.ptr);
        if (n > 1 and cap > 1 and self.ranks == null) {
            var sms: [round_streams]?mtp.Sampling = undefined;
            var masks: [round_streams][]const u32 = undefined;
            var outputs: [round_streams][]u32 = undefined;
            for (drafted[at..][0..n], 0..) |wi, i| {
                sms[i] = wins[wi].stream.sampling;
                masks[i] = wins[wi].masks;
                outputs[i] = out[wi].sampled[0..wins[wi].rows()];
            }
            try self.ops().drawsMany(self.logits.ptr, ds[0..n], sms[0..n], masks[0..n], outputs[0..n]);
        } else for (drafted[at..][0..n], ds[0..n]) |wi, d| try self.ops().draws(d, wins[wi].stream.sampling, wins[wi].masks, out[wi].sampled[0..wins[wi].rows()]);
        at += n;
    }
}

test {
    _ = @import("cuda_lanes_verify_test.zig");
}
