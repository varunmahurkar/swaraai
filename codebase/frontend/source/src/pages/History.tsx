import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import AudioPlayer from '../components/AudioPlayer';
import { api } from '../lib/api';
import type { HistoryItem, VoiceProfile } from '../types';

export default function History() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [profileFilter, setProfileFilter] = useState('');
  const [offset, setOffset] = useState(0);
  const limit = 20;

  const { data: profiles = [] } = useQuery<VoiceProfile[]>({
    queryKey: ['profiles'],
    queryFn: () => api.listProfiles(),
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ['history', { search, profileFilter, offset, limit }],
    queryFn: () =>
      api.getHistory({
        search: search || undefined,
        profile_id: profileFilter || undefined,
        limit,
        offset,
      }),
    placeholderData: (prev: import('../types').HistoryListResponse | undefined) => prev,
  });

  const items: HistoryItem[] = data?.items ?? [];
  const total: number = data?.total ?? 0;

  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this generation?')) return;
    setDeleting(id);
    try {
      await api.deleteGeneration(id);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    } finally {
      setDeleting(null);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setOffset(0);
  };

  const fmt = (iso: string) =>
    new Date(iso).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    });

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Generation History</h1>
        <p className="text-white/50 mt-1">Browse and play your previously generated audio</p>
      </div>

      {/* Filters */}
      <form onSubmit={handleSearch} className="flex gap-3 mb-6">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search text…"
          className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-white/30 focus:outline-none focus:border-violet-500 transition-colors"
        />
        <select
          value={profileFilter}
          onChange={(e) => { setProfileFilter(e.target.value); setOffset(0); }}
          className="bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500 transition-colors"
        >
          <option value="">All Profiles</option>
          {profiles.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <button
          type="submit"
          className="bg-violet-600 hover:bg-violet-500 text-white px-4 py-2 rounded-lg transition-colors text-sm"
        >
          Search
        </button>
      </form>

      {error instanceof Error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 mb-6 text-sm">
          {error.message}
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-16 text-white/40">Loading history…</div>
      ) : items.length === 0 ? (
        <div className="text-center py-16">
          <div className="text-5xl mb-4">📂</div>
          <p className="text-white/50">No generations found</p>
          {search && (
            <button
              onClick={() => { setSearch(''); setProfileFilter(''); }}
              className="mt-3 text-violet-400 hover:text-violet-300 text-sm"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <>
          <p className="text-sm text-white/40 mb-4">{total} generation{total !== 1 ? 's' : ''}</p>

          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.id}
                className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-white/20 transition-colors"
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-sm line-clamp-2">{item.text}</p>
                    <div className="flex flex-wrap gap-3 mt-1.5 text-xs text-white/40">
                      <span className="text-violet-400">{item.profile_name}</span>
                      <span>{item.language.toUpperCase()}</span>
                      <span>{item.duration.toFixed(1)}s</span>
                      <span>{fmt(item.created_at)}</span>
                      {item.instruct && <span className="italic">"{item.instruct}"</span>}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(item.id)}
                    disabled={deleting === item.id}
                    className="text-red-400/50 hover:text-red-400 disabled:opacity-40 transition-colors flex-shrink-0 text-sm"
                    title="Delete"
                  >
                    {deleting === item.id ? '…' : '🗑'}
                  </button>
                </div>
                <AudioPlayer
                  src={api.getGenerationAudioUrl(item.id)}
                  title={item.text.slice(0, 50)}
                />
              </div>
            ))}
          </div>

          {/* Pagination */}
          {total > limit && (
            <div className="flex items-center justify-center gap-4 mt-8">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:bg-white/5 disabled:opacity-40 transition-colors text-sm"
              >
                ← Previous
              </button>
              <span className="text-sm text-white/40">
                {offset + 1}–{Math.min(offset + limit, total)} of {total}
              </span>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
                className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:bg-white/5 disabled:opacity-40 transition-colors text-sm"
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
