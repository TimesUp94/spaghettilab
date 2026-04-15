import { useEffect, useRef, useState, useCallback, useMemo, memo } from "react";
import { X, ChevronDown, ChevronRight, Check, Loader2, ClipboardList } from "lucide-react";
import {
    QUESTIONNAIRE_TEMPLATES,
    DEFAULT_TEMPLATE_ID,
    META_SELECTED_TEMPLATE_KEY,
    getTemplateById,
    makeAnswerKey,
    type Section,
    type Question,
} from "../lib/questionnaireTemplate";
import { getQuestionnaireAnswers, setQuestionnaireAnswer } from "../api";

/**
 * Right-side slide-in panel for the FGC VOD-review questionnaire.
 * Desktop port of the web app's QuestionnairePanel — same DB table.
 */

interface Props {
    open: boolean;
    onClose: () => void;
    dbPath: string | null;
    replayId: string;
    readOnly?: boolean;
}

type SaveState = "idle" | "saving" | "saved";

export function QuestionnairePanel({ open, onClose, dbPath, replayId, readOnly = false }: Props) {
    const [selectedTemplateId, setSelectedTemplateId] = useState<string>(DEFAULT_TEMPLATE_ID);
    const [allAnswers, setAllAnswers] = useState<Record<string, string>>({});
    const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
    const [saveState, setSaveState] = useState<SaveState>("idle");
    const saveTimersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
    const hydratedRef = useRef(false);

    const template = useMemo(() => getTemplateById(selectedTemplateId), [selectedTemplateId]);

    // Hydrate once per open
    useEffect(() => {
        if (!open || !dbPath || !replayId || hydratedRef.current) return;
        (async () => {
            try {
                const stored = (await getQuestionnaireAnswers(dbPath, replayId)) || {};
                setAllAnswers(stored);
                const storedTemplate = stored[META_SELECTED_TEMPLATE_KEY];
                if (storedTemplate && QUESTIONNAIRE_TEMPLATES.some((t) => t.id === storedTemplate)) {
                    setSelectedTemplateId(storedTemplate);
                }
                hydratedRef.current = true;
            } catch (e) {
                console.error("Failed to hydrate questionnaire:", e);
            }
        })();
    }, [open, dbPath, replayId]);

    // Reset hydration when replay changes
    useEffect(() => {
        hydratedRef.current = false;
        setAllAnswers({});
    }, [replayId]);

    // Cleanup timers on unmount
    useEffect(() => () => {
        Object.values(saveTimersRef.current).forEach((id) => clearTimeout(id));
    }, []);

    const persist = useCallback(async (fullKey: string, value: string) => {
        if (!dbPath || !replayId) return;
        try {
            setSaveState("saving");
            await setQuestionnaireAnswer(dbPath, replayId, fullKey, value);
            setSaveState("saved");
            setTimeout(() => setSaveState((s) => (s === "saved" ? "idle" : s)), 1500);
        } catch (e) {
            console.error("Failed to save answer:", e);
            setSaveState("idle");
        }
    }, [dbPath, replayId]);

    const handleChange = useCallback((questionKey: string, value: string) => {
        const fullKey = makeAnswerKey(selectedTemplateId, questionKey);
        setAllAnswers((prev) => ({ ...prev, [fullKey]: value }));
        if (saveTimersRef.current[fullKey]) clearTimeout(saveTimersRef.current[fullKey]);
        saveTimersRef.current[fullKey] = setTimeout(() => persist(fullKey, value), 500);
    }, [selectedTemplateId, persist]);

    const handleBlur = useCallback((questionKey: string) => {
        const fullKey = makeAnswerKey(selectedTemplateId, questionKey);
        if (saveTimersRef.current[fullKey]) {
            clearTimeout(saveTimersRef.current[fullKey]);
            delete saveTimersRef.current[fullKey];
        }
        persist(fullKey, allAnswers[fullKey] || "");
    }, [selectedTemplateId, allAnswers, persist]);

    const handleKeyDown = useCallback((e: React.KeyboardEvent, questionKey: string) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            handleBlur(questionKey);
        }
    }, [handleBlur]);

    const handleTemplateChange = useCallback(async (newId: string) => {
        setSelectedTemplateId(newId);
        if (!readOnly && dbPath && replayId) {
            try {
                await setQuestionnaireAnswer(dbPath, replayId, META_SELECTED_TEMPLATE_KEY, newId);
                setAllAnswers((prev) => ({ ...prev, [META_SELECTED_TEMPLATE_KEY]: newId }));
            } catch (e) {
                console.error("Failed to persist template choice:", e);
            }
        }
    }, [readOnly, dbPath, replayId]);

    const toggleSection = useCallback((id: string) => {
        setCollapsed((prev) => ({ ...prev, [id]: !prev[id] }));
    }, []);

    if (!open) return null;

    return (
        <>
            <div
                onClick={onClose}
                className="fixed inset-0 z-30 bg-black/40 sm:hidden"
                aria-hidden="true"
            />

            <aside
                className="fixed right-0 top-0 h-full w-full sm:w-[420px] z-40 bg-gray-900 border-l border-gray-700 shadow-2xl flex flex-col"
                role="dialog"
                aria-label="Review Questionnaire"
            >
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-gray-900/95">
                    <div className="flex items-center gap-2 min-w-0">
                        <ClipboardList className="w-4 h-4 text-yellow-400 shrink-0" />
                        <h2 className="text-sm font-semibold text-white truncate">Review Questionnaire</h2>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white" title="Close">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Template selector */}
                <div className="px-4 py-3 border-b border-gray-700 bg-gray-800/50">
                    <label className="block text-xs font-semibold text-gray-400 mb-1.5 uppercase tracking-wider">
                        Game
                    </label>
                    <select
                        value={selectedTemplateId}
                        onChange={(e) => handleTemplateChange(e.target.value)}
                        disabled={readOnly}
                        className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:border-yellow-500 focus:ring-1 focus:ring-yellow-500 disabled:opacity-60 disabled:cursor-not-allowed"
                    >
                        {QUESTIONNAIRE_TEMPLATES.map((tpl) => (
                            <option key={tpl.id} value={tpl.id}>{tpl.label}</option>
                        ))}
                    </select>
                </div>

                <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-4 space-y-4">
                    {template.sections.map((section) => (
                        <SectionView
                            key={section.id}
                            section={section}
                            collapsed={!!collapsed[section.id]}
                            onToggle={() => toggleSection(section.id)}
                            getValue={(questionKey) => allAnswers[makeAnswerKey(selectedTemplateId, questionKey)] || ""}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            onKeyDown={handleKeyDown}
                            readOnly={readOnly}
                        />
                    ))}
                </div>

                <div className="border-t border-gray-700 bg-gray-900/95 px-4 py-2 flex items-center gap-2 text-xs">
                    {saveState === "saving" && (
                        <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-yellow-400" />
                            <span className="text-gray-400">Saving...</span>
                        </>
                    )}
                    {saveState === "saved" && (
                        <>
                            <Check className="w-3.5 h-3.5 text-green-400" />
                            <span className="text-gray-400">Saved</span>
                        </>
                    )}
                    {saveState === "idle" && (
                        <span className="text-gray-500 italic">
                            {readOnly ? "Read-only view" : "Answers save automatically"}
                        </span>
                    )}
                </div>
            </aside>
        </>
    );
}

interface SectionProps {
    section: Section;
    collapsed: boolean;
    onToggle: () => void;
    getValue: (questionKey: string) => string;
    onChange: (questionKey: string, value: string) => void;
    onBlur: (questionKey: string) => void;
    onKeyDown: (e: React.KeyboardEvent, questionKey: string) => void;
    readOnly: boolean;
}

const SectionView = memo(function SectionView({ section, collapsed, onToggle, getValue, onChange, onBlur, onKeyDown, readOnly }: SectionProps) {
    return (
        <section className="bg-gray-800/60 border border-gray-700 rounded-lg overflow-hidden">
            <button
                onClick={onToggle}
                className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-700/50 transition-colors"
            >
                <span className="text-xs font-bold text-yellow-400 uppercase tracking-wider">{section.title}</span>
                {collapsed ? <ChevronRight className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
            </button>
            {!collapsed && (
                <div className="px-3 pb-3 pt-1 space-y-3">
                    {section.questions.map((q) => (
                        <QuestionRow
                            key={q.key}
                            question={q}
                            value={getValue(q.key)}
                            onChange={(v) => onChange(q.key, v)}
                            onBlur={() => onBlur(q.key)}
                            onKeyDown={(e) => onKeyDown(e, q.key)}
                            readOnly={readOnly}
                        />
                    ))}
                </div>
            )}
        </section>
    );
});

interface QuestionProps {
    question: Question;
    value: string;
    onChange: (value: string) => void;
    onBlur: () => void;
    onKeyDown: (e: React.KeyboardEvent) => void;
    readOnly: boolean;
}

const QuestionRow = memo(function QuestionRow({ question, value, onChange, onBlur, onKeyDown, readOnly }: QuestionProps) {
    const rows = question.rows || 2;
    return (
        <div>
            <label className="block text-xs text-gray-300 leading-snug mb-1.5 whitespace-pre-line">{question.label}</label>
            {readOnly ? (
                <p className="text-sm text-gray-200 whitespace-pre-wrap min-h-[1.5rem]">
                    {value || <span className="text-gray-500 italic">—</span>}
                </p>
            ) : (
                <textarea
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onBlur={onBlur}
                    onKeyDown={onKeyDown}
                    rows={rows}
                    placeholder={question.placeholder || ""}
                    className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-yellow-500 focus:ring-1 focus:ring-yellow-500 resize-y"
                />
            )}
        </div>
    );
});
