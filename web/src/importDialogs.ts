// The dialogs that import a voice another tool made (engines with capabilities.voice_import):
// each engine's own form, sent to POST /api/loras/import/<engine>. A new engine adds its entry here.
import type { ComponentType } from 'react'
import { GptSovitsImportDialog } from './components/GptSovitsImportDialog'
import { RvcImportDialog } from './components/RvcImportDialog'
import type { Messages } from './i18n'

export interface ImportDialogProps {
  onImported: (id: string, engine: string) => void
  onError: (message: string) => void
  onClose: () => void
}

export interface ImportDialog {
  Dialog: ComponentType<ImportDialogProps>
  label: (t: Messages) => string // its button
  title?: (t: Messages) => string
}

export const IMPORT_DIALOGS: Record<string, ImportDialog> = {
  gpt_sovits: {
    Dialog: GptSovitsImportDialog,
    label: (t) => t.gsvImportButton,
    title: (t) => t.gsvImportButtonTitle,
  },
  rvc: { Dialog: RvcImportDialog, label: (t) => t.rvcImportButton },
}

/** The engine's import dialog, if it imports voices from another tool. */
export function importDialog(engine: { id: string; capabilities: { voice_import: boolean } } | null) {
  return engine?.capabilities.voice_import ? (IMPORT_DIALOGS[engine.id] ?? null) : null
}
