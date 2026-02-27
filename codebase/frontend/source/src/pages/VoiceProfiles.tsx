import { useState } from 'react';
import { useVoiceProfiles, useProfileSamples } from '../hooks/useVoiceProfiles';
import ProfileForm from '../components/ProfileForm';
import SampleUpload from '../components/SampleUpload';
import { api } from '../lib/api';
import type { VoiceProfile, ProfileSample } from '../types';

function SamplesList({ profileId }: { profileId: string }) {
  const { data: samples = [], isLoading } = useProfileSamples(profileId);
  const { deleteSample } = useVoiceProfiles();
  const [deleting, setDeleting] = useState<string | null>(null);

  if (isLoading) return <p className="text-white/40 text-sm">Loading samples…</p>;
  if (!samples.length) return <p className="text-white/40 text-sm italic">No samples yet.</p>;

  return (
    <ul className="space-y-2">
      {samples.map((s: ProfileSample) => (
        <li key={s.id} className="flex items-start gap-2 bg-white/5 rounded-lg px-3 py-2">
          <div className="flex-1 min-w-0">
            <a
              href={api.getSampleAudioUrl(s.id)}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-violet-400 hover:text-violet-300 block truncate"
            >
              🔊 Sample audio
            </a>
            <p className="text-xs text-white/60 mt-0.5 line-clamp-2">{s.reference_text}</p>
          </div>
          <button
            onClick={async () => {
              setDeleting(s.id);
              await deleteSample(s.id, s.profile_id);
              setDeleting(null);
            }}
            disabled={deleting === s.id}
            className="text-xs text-red-400/60 hover:text-red-400 disabled:opacity-40 flex-shrink-0"
          >
            {deleting === s.id ? '…' : '✕'}
          </button>
        </li>
      ))}
    </ul>
  );
}

function ProfileCard({ profile, onDeleted }: { profile: VoiceProfile; onDeleted: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const [showAddSample, setShowAddSample] = useState(false);
  const [showEdit, setShowEdit] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const { deleteProfile, updateProfile, addSample } = useVoiceProfiles();

  const langEmoji: Record<string, string> = {
    en: '🇬🇧', zh: '🇨🇳', ja: '🇯🇵', ko: '🇰🇷',
    de: '🇩🇪', fr: '🇫🇷', es: '🇪🇸', it: '🇮🇹', pt: '🇵🇹', ru: '🇷🇺',
  };

  const handleDelete = async () => {
    if (!confirm(`Delete profile "${profile.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    await deleteProfile(profile.id);
    onDeleted();
  };

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:border-white/20 transition-colors">
      {/* Card header */}
      <div
        className="px-4 py-3 flex items-center gap-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center text-xl flex-shrink-0">
          {profile.avatar_path ? (
            <img
              src={api.getAvatarUrl(profile.id)}
              alt={profile.name}
              className="w-full h-full object-cover rounded-xl"
            />
          ) : (
            '🎤'
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-white truncate">{profile.name}</span>
            <span className="text-base">{langEmoji[profile.language] ?? '🌐'}</span>
          </div>
          {profile.description && (
            <p className="text-xs text-white/50 truncate">{profile.description}</p>
          )}
        </div>
        <span className="text-white/30 text-sm">{expanded ? '▲' : '▼'}</span>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-white/5 space-y-4">
          {showEdit ? (
            <div className="pt-3">
              <ProfileForm
                initialValues={{ name: profile.name, description: profile.description ?? '', language: profile.language }}
                onSubmit={async (data) => {
                  await updateProfile(profile.id, data);
                  setShowEdit(false);
                }}
                onCancel={() => setShowEdit(false)}
              />
            </div>
          ) : showAddSample ? (
            <div className="pt-3">
              <SampleUpload
                profileId={profile.id}
                onAdd={(audio, referenceText) => addSample(profile.id, audio, referenceText)}
                onDone={() => setShowAddSample(false)}
              />
            </div>
          ) : (
            <>
              <div className="pt-3">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-white/70">Voice Samples</h4>
                  <button
                    onClick={() => setShowAddSample(true)}
                    className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
                  >
                    + Add Sample
                  </button>
                </div>
                <SamplesList profileId={profile.id} />
              </div>

              <div className="flex gap-2 pt-2 border-t border-white/5">
                <button
                  onClick={() => setShowEdit(true)}
                  className="text-xs text-white/50 hover:text-white transition-colors px-3 py-1.5 rounded-lg hover:bg-white/5"
                >
                  Edit
                </button>
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="text-xs text-red-400/60 hover:text-red-400 disabled:opacity-40 transition-colors px-3 py-1.5 rounded-lg hover:bg-red-400/10"
                >
                  {deleting ? 'Deleting…' : 'Delete'}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function VoiceProfiles() {
  const { profiles, loading, error, createProfile, refetch } = useVoiceProfiles();
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white">Voice Profiles</h1>
          <p className="text-white/50 mt-1">Manage your voice cloning profiles and samples</p>
        </div>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="bg-violet-600 hover:bg-violet-500 text-white font-medium px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
        >
          <span>+</span> New Profile
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 mb-6 text-sm">
          {error}
        </div>
      )}

      {showCreate && (
        <div className="bg-white/5 border border-white/10 rounded-xl p-5 mb-6">
          <h3 className="text-lg font-medium text-white mb-4">Create New Profile</h3>
          <ProfileForm
            onSubmit={async (data) => {
              await createProfile(data);
              setShowCreate(false);
            }}
            onCancel={() => setShowCreate(false)}
          />
        </div>
      )}

      {loading ? (
        <div className="text-center py-16 text-white/40">Loading profiles…</div>
      ) : profiles.length === 0 ? (
        <div className="text-center py-16">
          <div className="text-6xl mb-4">🎤</div>
          <h3 className="text-xl font-medium text-white/70 mb-2">No profiles yet</h3>
          <p className="text-white/40 text-sm mb-6">Create your first voice profile to get started</p>
          <button
            onClick={() => setShowCreate(true)}
            className="bg-violet-600 hover:bg-violet-500 text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
          >
            Create Profile
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {profiles.map((profile: VoiceProfile) => (
            <ProfileCard key={profile.id} profile={profile} onDeleted={refetch} />
          ))}
        </div>
      )}
    </div>
  );
}
