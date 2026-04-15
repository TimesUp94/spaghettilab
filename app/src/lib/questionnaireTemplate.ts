/**
 * Questionnaire templates for the VOD-review checklist (desktop app).
 *
 * Keep this file in sync with the web app at
 * FGCScoreboard/app/components/spaghetti-lab/questionnaireTemplate.js —
 * both apps share the same SQLite `questionnaire_answers` schema so files
 * round-trip cleanly.
 *
 * Answers are stored in the DB keyed by `${template.id}.${question.key}`
 * so multiple games' answers never collide. Template and question ids must
 * match the web app exactly.
 */

export type Question = {
    key: string
    label: string
    placeholder?: string
    rows?: number
}

export type Section = {
    id: string
    title: string
    questions: Question[]
}

export type Template = {
    id: string
    label: string
    sections: Section[]
}

// ── Guilty Gear: Strive ────────────────────────────────────────────────────
const GGST_TEMPLATE: Template = {
    id: 'ggst',
    label: 'Guilty Gear: Strive',
    sections: [
        {
            id: 'general',
            title: 'GENERAL',
            questions: [
                {
                    key: 'general.thoughts',
                    label: 'General thoughts',
                    placeholder: 'Write your overall impressions of the match...',
                    rows: 5,
                },
            ],
        },
        {
            id: 'roundstart',
            title: 'ROUNDSTART',
            questions: [
                { key: 'roundstart.main_options', label: "Do I know 1 or 2 main options for my opponent's character?" },
                { key: 'roundstart.dry_options',  label: 'Are they using "dry" options (buttons, moves, jump, backdash)?' },
                { key: 'roundstart.fluid_options', label: 'Or more fluid options (walk forward > 2K / walk back > F.S)?' },
            ],
        },
        {
            id: 'neutral',
            title: 'NEUTRAL',
            questions: [
                { key: 'neutral.reactive_vs_proactive', label: 'Am I playing reactive/passive or proactive/aggressive?' },
                { key: 'neutral.not_reacted_or_predicted', label: "If I got hit, was it because I didn't react, or because I didn't predict it?" },
                { key: 'neutral.too_many_airdashes', label: 'Am I doing too many airdashes (forward)?' },
                { key: 'neutral.gapclosing', label: 'Do I have options for hard gapclosing (neutral skip)? If so, am I alternating them well?' },
                { key: 'neutral.dashblock', label: 'Am I dashblocking well?' },
                { key: 'neutral.anti_air', label: 'Am I anti-airing?' },
                { key: 'neutral.my_strong_range', label: 'Where (at what range) am I strong?' },
                { key: 'neutral.opp_strong_range', label: 'Where (at what range) is my opponent strong? Can I zone them, or are they zoning me?' },
                { key: 'neutral.meter_usage', label: 'Am I using meter well, or am I ending the round with a full meter?' },
            ],
        },
        {
            id: 'defense',
            title: 'DEFENSE',
            questions: [
                { key: 'defense.fd', label: 'Am I pressing FD?' },
                { key: 'defense.mashing', label: 'Am I mashing randomly, or following the universal string rules (C.S > 2S > 5H > special)?' },
                { key: 'defense.reactable_resets', label: 'Does the opponent have pressure resets that are reactable on startup? If so, am I reacting?' },
                { key: 'defense.backdash_spam', label: 'Am I spamming backdash too much? If so, in which situations?' },
                { key: 'defense.throw_spam', label: 'Am I spamming throws too much? If so, in which situations?' },
                { key: 'defense.knockdown_rps', label: 'Opponent forces me into an RPS (knockdown): what are their options? What are mine?' },
                { key: 'defense.rps_risk_reward', label: 'I took an option inside the RPS: did I evaluate the risk/reward well?' },
            ],
        },
        {
            id: 'offense',
            title: 'OFFENSE',
            questions: [
                { key: 'offense.gatling_delay', label: 'Am I delaying my gatlings, or dumping everything with the same timing?' },
                { key: 'offense.oki_after_knockdown', label: 'If I get a knockdown, do I close in and take oki, or let it go? (Always take oki)' },
                { key: 'offense.meaties', label: 'Am I landing meaties well, or is my opponent interrupting them? If so, with what?' },
                { key: 'offense.stagger_pressure', label: 'Am I doing stagger pressure well from C.S?' },
                { key: 'offense.dp_handling', label: 'If my opponent has a DP (or invincible reversal), am I playing around it well?' },
                { key: 'offense.safejumps', label: 'Do I have options to beat reversals without baiting them (safejump)? Am I using them well?' },
                { key: 'offense.fd_spacing_traps', label: 'Do I have options against an opponent who presses FD (moves whose FD pushback lets me set a spacing trap)?' },
                { key: 'offense.opp_defense_breaks', label: 'How are they holding up on defense? When does their defense break?' },
                { key: 'offense.impatient_or_not', label: 'Are they impatient? Am I punishing them accordingly (reset/throw if they respect, frametrap by delaying gatlings if not)?' },
            ],
        },
    ],
}

export const QUESTIONNAIRE_TEMPLATES: Template[] = [
    GGST_TEMPLATE,
    // Add future templates here (same shape).
]

export const DEFAULT_TEMPLATE_ID = 'ggst'

// Reserved key for remembering which template a replay uses.
// Leading underscore keeps it out of any template's namespace.
export const META_SELECTED_TEMPLATE_KEY = '_meta.selected_template'

export function getTemplateById(id: string): Template {
    return QUESTIONNAIRE_TEMPLATES.find((t) => t.id === id) || QUESTIONNAIRE_TEMPLATES[0]
}

export function makeAnswerKey(templateId: string, questionKey: string): string {
    return `${templateId}.${questionKey}`
}
