# Security and Maintenance Concerns

## Security Issues
- API keys stored in `alpha_keys.json` (should be in .env)
- Conversation logs may contain sensitive data
- No apparent rate limiting beyond auth module

## Maintenance Issues
- No automated testing in CI/CD
- No code coverage metrics
- Inconsistent error handling patterns
- Missing logging in some modules
