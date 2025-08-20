function ask_pulsePal(query)
    % ASK_PULSPAL Send MATLAB code and query to PulsePal
    %   ask_pulsePal(query) - Sends your query along with current editor code
    %
    % Usage:
    %   ask_pulsePal('Why is my gradient exceeding limits?')
    %   ask_pulsePal('Help me debug this spin echo sequence')
    %
    % Session Management:
    %   For best results, open PulsePal in your browser FIRST, then use this function
    %   to send code to your existing session. This preserves conversation context.
    %
    % The function will:
    %   1. Get the code from your active MATLAB editor
    %   2. Format it with your query
    %   3. Copy to clipboard
    %   4. Guide you to paste in your existing PulsePal session

    % Check input
    if nargin < 1 || isempty(query)
        error('Please provide a query. Example: ask_pulsePal(''Why is my RF pulse not working?'')');
    end

    if ~ischar(query) && ~isstring(query)
        error('Query must be a string');
    end

    % Convert to char if needed
    query = char(query);

    % Load configuration
    try
        config = pulsePal_config();
    catch
        % Fallback if config file is missing
        config = struct();
        config.url = 'http://localhost:8000';
        config.use_local = true;
        config.session_mode = 'existing';
        config.auto_open_browser = false;
        config.show_session_instructions = true;
        config.verbose = true;
    end

    % Check if PulsePal is accessible (for local mode)
    if config.use_local && config.verbose
        try
            % Simple check if localhost is responding
            [~, status] = urlread(config.url);
            if ~status
                warning('Cannot connect to local PulsePal at %s', config.url);
                fprintf('Make sure Chainlit is running: chainlit run chainlit_app.py\n');
            end
        catch
            % urlread might not be available or fail
            if config.verbose
                fprintf('📍 Using local URL: %s\n', config.url);
                fprintf('   Make sure Chainlit is running locally.\n');
            end
        end
    end

    % Get active editor content
    try
        editor = matlab.desktop.editor.getActive;
        if isempty(editor)
            warning('No active MATLAB editor found.');
            warning('To include your code, open a .m file in the MATLAB editor first.');

            % Just copy the query
            clipboard('copy', query);

            % Provide session-aware instructions
            print_session_instructions(config, false);
            return;
        end
    catch ME
        % Editor API might not be available in some MATLAB versions
        warning('Could not access MATLAB editor: %s', ME.message);

        % Just copy the query
        clipboard('copy', query);

        % Provide session-aware instructions
        print_session_instructions(config, false);
        return;
    end

    % Get code and filename
    code = editor.Text;
    filename = editor.Filename;

    if isempty(filename)
        % Untitled file
        [~, name, ext] = fileparts('untitled.m');
    else
        [~, name, ext] = fileparts(filename);
    end

    % Check if code is too large (>1MB)
    if length(code) > 1024*1024
        warning('Code is too large (>1MB). Consider selecting specific portions.');
        response = input('Continue anyway? (y/n): ', 's');
        if ~strcmpi(response, 'y')
            fprintf('Operation cancelled.\n');
            return;
        end
    end

    % Format the message
    formatted_message = sprintf(['%s\n\n' ...
        '--- Code from %s%s ---\n' ...
        '%s\n' ...
        '--- End of %s%s ---'], ...
        query, name, ext, code, name, ext);

    % Copy to clipboard
    clipboard('copy', formatted_message);

    % Provide session-aware instructions
    print_session_instructions(config, true, query, name, ext, code);
end

function print_session_instructions(config, has_code, query, name, ext, code)
    % Helper function to print session-aware instructions

    fprintf('\n=====================================\n');
    if has_code
        fprintf('✅ Your code and query have been copied to clipboard!\n');
    else
        fprintf('✅ Your query has been copied to clipboard!\n');
    end
    fprintf('=====================================\n\n');

    if strcmp(config.session_mode, 'existing')
        % User should use existing browser session
        fprintf('📌 SESSION MANAGEMENT:\n');
        fprintf('   1. Open PulsePal in your browser if not already open:\n');
        fprintf('      %s\n', config.url);
        if config.use_local
            fprintf('      (Make sure Chainlit is running locally first)\n');
        end
        fprintf('   2. If using production, log in with your API key\n');
        fprintf('   3. Paste your %s in the chat (Ctrl+V or Cmd+V)\n', ...
            ternary(has_code, 'code and query', 'query'));
        fprintf('   4. Your existing conversation context will be preserved\n\n');

        if config.auto_open_browser
            fprintf('🌐 Opening browser (if needed)...\n');
            web(config.url, '-browser');
        else
            fprintf('💡 Tip: Keep your browser session open for continuous work\n');
        end
    else
        % Legacy behavior - open new browser session
        fprintf('🌐 Opening PulsePal in new browser window...\n');
        web(config.url, '-browser');
        fprintf('📋 Just paste (Ctrl+V or Cmd+V) in the PulsePal chat window.\n');
    end

    if has_code && nargin >= 6
        fprintf('\n📝 Query: "%s"\n', query);
        fprintf('📄 Code from: %s%s (%d lines)\n', name, ext, length(strfind(code, newline)) + 1);
    elseif ~has_code && nargin >= 3
        fprintf('\n📝 Query: "%s"\n', query);
    end

    fprintf('\n💡 TIPS:\n');
    if has_code
        fprintf('   - Your code will be automatically formatted\n');
        fprintf('   - PulsePal will analyze your code and provide specific help\n');
    end
    fprintf('   - You can also drag and drop .m files directly into the chat\n');
    if strcmp(config.session_mode, 'existing')
        fprintf('   - Your conversation history is preserved across queries\n');
        fprintf('   - Keep the same browser tab open for best results\n');
    end
    fprintf('\n');
end

function result = ternary(condition, true_val, false_val)
    % Simple ternary operator helper
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
