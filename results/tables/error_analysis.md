# Error Analysis

## Error Categories (sampled 50 errors)
| Category | Count |
|----------|-------|
| Multi-word aspect boundary | 6 |
| Multiple aspects contrasting | 3 |
| Negation scope | 3 |
| Sarcasm | 2 |
| Other | 2 |
| Annotation noise | 2 |
| Implicit aspect | 1 |

## Selected Examples
**Sentence**: I was given a demonstration of Windows 8.
**Aspect**: Windows 8 | **True**: None | **Pred**: neutral
**Category**: Sarcasm

**Sentence**: Unfortunately, it runs XP and Microsoft is dropping support next April.
**Aspect**: XP | **True**: None | **Pred**: positive
**Category**: Other

**Sentence**: This thing is awesome, everything always works, everything is always easy to set up, everything is compatible, its literally everything I could ask for.
**Aspect**: works | **True**: None | **Pred**: neutral
**Category**: Multiple aspects contrasting

**Sentence**: It feels cheap, the keyboard is not very sensitive.
**Aspect**: keyboard | **True**: None | **Pred**: neutral
**Category**: Negation scope

**Sentence**: It suddenly can not work.
**Aspect**: work | **True**: None | **Pred**: positive
**Category**: Multi-word aspect boundary

