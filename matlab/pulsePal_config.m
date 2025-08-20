function config = pulsePal_config()
    % PULSPAL_CONFIG Configuration settings for PulsePal MATLAB integration
    %   config = pulsePal_config() returns configuration struct
    %
    % Configuration includes:
    %   - URL (local or production)
    %   - Session management preferences
    %   - Browser behavior

    % Default configuration
    config = struct();

    % URL Configuration
    % Change this to switch between local and production
    config.use_local = true;  % Set to false for production

    if config.use_local
        config.url = 'http://localhost:8000';
    else
        config.url = 'https://pulsepal.up.railway.app';
    end

    % Session Management
    % How should MATLAB interact with browser sessions?
    config.session_mode = 'existing';  % Options: 'existing' or 'new'
    % 'existing' - User should have browser session open first
    % 'new' - MATLAB opens new browser session (legacy behavior)

    % Browser Behavior
    config.auto_open_browser = false;  % Don't auto-open if using existing session
    config.show_session_instructions = true;  % Show instructions for session management

    % Development/Debug Settings
    config.verbose = true;  % Show detailed messages
    config.check_session = true;  % Check if localhost is running (for local mode)

    % File Size Limits
    config.max_file_size_mb = 1;  % Maximum file size in MB
    config.warn_large_files = true;  % Warn before sending large files

    % Clipboard Settings
    config.format_code_blocks = true;  % Add markdown code blocks
    config.include_filename = true;  % Include filename in formatted message

    % Display current configuration
    if config.verbose
        fprintf('\n📋 PulsePal Configuration:\n');
        fprintf('   Mode: %s\n', ternary(config.use_local, 'LOCAL TESTING', 'PRODUCTION'));
        fprintf('   URL: %s\n', config.url);
        fprintf('   Session: %s\n', config.session_mode);
        fprintf('   Auto-open browser: %s\n', ternary(config.auto_open_browser, 'Yes', 'No'));
        fprintf('\n');
    end
end

function result = ternary(condition, true_val, false_val)
    % Simple ternary operator helper
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
