import { useMemo, useState } from 'react'
import { DEFAULT_ENGINE } from '../api'
import type { Engine, Lora } from '../api'
import type { VoiceChoice } from '../engines'
import { useT } from '../i18n'
import { groupVoices } from '../voices'
import { ExportDialog } from './ExportDialog'
import { Info } from './Info'
import { ReferenceVoiceDialog } from './ReferenceVoiceDialog'
import { VoiceReference } from './VoiceReference'

interface Props {
  engines: Engine[]
  loras: Lora[]
  selected: VoiceChoice
  modelId: string | null
  onSelect: (choice: VoiceChoice) => void
  onRefresh: () => void
  onImported: (id: string, engine: string) => void // a saved-reference voice made here
  onError: (message: string) => void
}

/** What the chosen voice is, at the top of the settings: its checkpoint, its own reference clip,
 * Export, or (on a base model that can) Save as a voice. */
export function VoiceDetails({ engines, loras, selected, modelId, onSelect, onRefresh, onImported, onError }: Props) {
  const t = useT()
  const voices = useMemo(() => groupVoices(loras, t), [loras, t])
  const [exporting, setExporting] = useState<Lora | null>(null)
  const [savingVoice, setSavingVoice] = useState<Engine | null>(null)
  const engineOf = (id: string) => engines.find((e) => e.id === id) ?? null
  const engine = engineOf(selected.engine)
  const current = loras.find((l) => l.id === selected.lora) ?? null
  const currentVoice = current ? voices.find((v) => v.key === current.voice_path) : null
  const referenceOnly = !!engine?.capabilities.reference_voices
  // A hidden duplicate (e.g. picked from the Train tab) still needs an option.
  const options =
    currentVoice && current && !currentVoice.checkpoints.some((c) => c.lora.id === current.id)
      ? [...currentVoice.checkpoints, { lora: current, step: current.step, label: current.id.split('/').at(-1)! }]
      : (currentVoice?.checkpoints ?? [])

  const layout = (lora: Lora) =>
    lora.version
      ? lora.version
      : lora.config
        ? t.loraLayout(
            lora.config.r,
            lora.config.alpha,
            [lora.config.enable_lm && 'LM', lora.config.enable_dit && 'DiT', lora.config.enable_proj && t.projections]
              .filter(Boolean)
              .join(' + '),
          )
        : lora.rank !== null && lora.alpha !== null
          ? t.rankAlpha(lora.rank, lora.alpha)
          : t.defaultLayout

  return (
    <section className="settings-section voice-details">
      {current && currentVoice ? (
        <>
          <div className="settings-head">
            <h3>
              {t.voiceKind(referenceOnly)}
              <Info
                text={[
                  engine?.name ?? current.engine,
                  referenceOnly ? t.savedReference : layout(current),
                  t.baseOf(current.base_model_id ?? current.base_model ?? t.notRecorded),
                ].join(' · ')}
              />
            </h3>
            {!!engine?.capabilities.package && (
              <button type="button" className="ghost small" onClick={() => setExporting(current)}>
                {t.export}
              </button>
            )}
          </div>
          {options.length > 1 && (
            <label className="field">
              <span>{t.checkpoint}</span>
              <select
                value={current.id}
                onChange={(e) => onSelect({ engine: current.engine, lora: e.target.value })}
                aria-label={t.checkpoint}
              >
                {options.map((c, i) => (
                  <option key={c.lora.id} value={c.lora.id}>
                    {c.label}
                    {i === 0 ? t.final : ''}
                  </option>
                ))}
              </select>
            </label>
          )}
          {!!engine?.capabilities.voice_reference && (
            <VoiceReference
              key={current.id} // another voice: a fresh form (not the last voice's choices)
              lora={current}
              count={engine.capabilities.voice_reference_clips ?? 1}
              search={engine.capabilities.reference_search}
              onChanged={onRefresh}
              onError={onError}
            />
          )}
        </>
      ) : (
        <div className="settings-head">
          <h3>
            {t.baseModel}
            <Info
              text={[engine?.name ?? selected.engine, t.zeroShotInfo, selected.engine === DEFAULT_ENGINE && modelId]
                .filter(Boolean)
                .join(' · ')}
            />
          </h3>
          {engine?.capabilities.reference_voices && !engine.unavailable && (
            <button type="button" className="ghost small" onClick={() => setSavingVoice(engine)}>
              {t.saveAsVoice}
            </button>
          )}
        </div>
      )}

      {savingVoice && (
        <ReferenceVoiceDialog
          engine={savingVoice}
          onCreated={onImported}
          onError={onError}
          onClose={() => setSavingVoice(null)}
        />
      )}
      {exporting && (
        <ExportDialog
          lora={exporting}
          engine={engineOf(exporting.engine)}
          final={currentVoice?.checkpoints[0]?.lora.id === exporting.id}
          onError={onError}
          onClose={() => setExporting(null)}
        />
      )}
    </section>
  )
}
