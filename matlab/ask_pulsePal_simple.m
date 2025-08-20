function ask_pulsePal_simple(query)
    % ASK_PULSPAL_SIMPLE Send only query to PulsePal (no code)
    %   ask_pulsePal_simple(query) - Copies your query for existing PulsePal session
    %
    % Usage:
    %   ask_pulsePal_simple('What is T1 relaxation?')
    %   ask_pulsePal_simple('How does k-space work?')
    %
    % Session Management:
    %   For best results, open PulsePal in your browser FIRST, then use this function
    %   to send queries to your existing session. This preserves conversation context.
    %
    % This is the simple version that doesn't include any code,
    % just your question. Use ask_pulsePal() to include code.

    % Check input
    if nargin < 1 || isempty(query)
        error('Please provide a query. Example: ask_pulsePal_simple(''What is T1 relaxation?'')');
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

    % Copy query to clipboard
    clipboard('copy', query);

    % Instructions
    fprintf('\n=====================================\n');
    fprintf('✅ Query copied to clipboard!\n');
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
        fprintf('   3. Paste your query in the chat (Ctrl+V or Cmd+V)\n');
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
        fprintf('📋 Paste (Ctrl+V or Cmd+V) in PulsePal chat window.\n');
    end

    fprintf('\n📝 Query: "%s"\n', query);

    % Tips
    fprintf('\n💡 TIPS:\n');
    fprintf('   - Use ask_pulsePal() to include your code for debugging\n');
    fprintf('   - You can also drag and drop .m files directly into the chat\n');
    if strcmp(config.session_mode, 'existing')
        fprintf('   - Your conversation history is preserved across queries\n');
        fprintf('   - Keep the same browser tab open for best results\n');
    end
    fprintf('\n');
end
