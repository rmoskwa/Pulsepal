function install_pulsePal_matlab()
    % INSTALL_PULSPAL_MATLAB Set up PulsePal MATLAB integration
    %   install_pulsePal_matlab() - Adds PulsePal helpers to MATLAB path
    %
    % This function will:
    %   1. Add the PulsePal MATLAB functions to your path
    %   2. Save the path for future sessions
    %   3. Test the installation
    %   4. Show usage examples
    
    fprintf('=====================================\n');
    fprintf('🧠 Installing PulsePal MATLAB Helper\n');
    fprintf('=====================================\n\n');
    
    % Get installation directory
    current_dir = fileparts(mfilename('fullpath'));
    
    if isempty(current_dir)
        error('Could not determine installation directory. Please run from the matlab folder.');
    end
    
    fprintf('📁 Installation directory: %s\n', current_dir);
    
    % Check if functions exist
    files_to_check = {'ask_pulsePal.m', 'ask_pulsePal_simple.m', 'pulsePal_config.m'};
    missing_files = {};
    
    for i = 1:length(files_to_check)
        filepath = fullfile(current_dir, files_to_check{i});
        if ~exist(filepath, 'file')
            missing_files{end+1} = files_to_check{i};
        end
    end
    
    if ~isempty(missing_files)
        error('Missing files: %s\nPlease ensure all files are in the matlab folder.', strjoin(missing_files, ', '));
    end
    
    % Add to MATLAB path
    fprintf('📂 Adding to MATLAB path...\n');
    addpath(current_dir);
    
    % Try to save path (might fail if user doesn't have permissions)
    try
        savepath;
        fprintf('✅ Path saved permanently\n');
    catch ME
        warning('Could not save path permanently: %s', ME.message);
        fprintf('⚠️ Path added for this session only.\n');
        fprintf('   To make permanent, run: savepath\n');
    end
    
    % Test the installation
    fprintf('\n📋 Testing installation...\n');
    
    % Test 1: Check if functions are accessible
    if exist('ask_pulsePal', 'file') == 2
        fprintf('  ✅ ask_pulsePal function found\n');
    else
        error('  ❌ ask_pulsePal function not found');
    end
    
    if exist('ask_pulsePal_simple', 'file') == 2
        fprintf('  ✅ ask_pulsePal_simple function found\n');
    else
        error('  ❌ ask_pulsePal_simple function not found');
    end
    
    % Test 2: Check clipboard access
    try
        clipboard('copy', 'test');
        clipboard_test = clipboard('paste');
        if strcmp(clipboard_test, 'test')
            fprintf('  ✅ Clipboard access working\n');
        else
            warning('  ⚠️ Clipboard may not be working correctly');
        end
    catch
        warning('  ⚠️ Could not test clipboard functionality');
    end
    
    % Configuration setup
    fprintf('\n📋 Configuring PulsePal settings...\n');
    
    % Ask user about their setup
    fprintf('\nHow will you use PulsePal?\n');
    fprintf('1. Local testing (http://localhost:8000)\n');
    fprintf('2. Production (https://pulsepal.up.railway.app)\n');
    
    choice = input('Enter your choice (1 or 2) [default: 1]: ', 's');
    if isempty(choice)
        choice = '1';
    end
    
    % Update configuration file
    config_file = fullfile(current_dir, 'pulsePal_config.m');
    if strcmp(choice, '2')
        fprintf('Configuring for PRODUCTION use...\n');
        % Read current config and update
        try
            config_content = fileread(config_file);
            config_content = strrep(config_content, 'config.use_local = true;', 'config.use_local = false;');
            
            % Write updated config
            fid = fopen(config_file, 'w');
            fprintf(fid, '%s', config_content);
            fclose(fid);
            fprintf('✅ Configuration updated for production\n');
        catch
            warning('Could not update configuration file automatically');
            fprintf('Please edit pulsePal_config.m and set: config.use_local = false\n');
        end
    else
        fprintf('✅ Configured for LOCAL testing\n');
        fprintf('   URL: http://localhost:8000\n');
        fprintf('   Remember to run: chainlit run chainlit_app.py\n');
    end
    
    % Installation complete
    fprintf('\n=====================================\n');
    fprintf('✅ Installation complete!\n');
    fprintf('=====================================\n\n');
    
    % Show usage instructions
    fprintf('📚 USAGE INSTRUCTIONS:\n\n');
    
    fprintf('🔄 RECOMMENDED WORKFLOW (Session Management):\n');
    fprintf('   1. Open PulsePal in your browser FIRST\n');
    if strcmp(choice, '1')
        fprintf('      http://localhost:8000\n');
    else
        fprintf('      https://pulsepal.up.railway.app\n');
        fprintf('      (Log in with your API key if required)\n');
    end
    fprintf('   2. Keep this browser tab open throughout your work\n');
    fprintf('   3. Use MATLAB functions to send queries to your session\n\n');
    
    fprintf('1️⃣ To send code with a question:\n');
    fprintf('   >> ask_pulsePal(''Why is my gradient exceeding limits?'')\n');
    fprintf('   This will:\n');
    fprintf('   - Get code from your active MATLAB editor\n');
    fprintf('   - Copy it with your question to clipboard\n');
    fprintf('   - Guide you to paste in your EXISTING browser session\n');
    fprintf('   - Preserve your conversation context\n\n');
    
    fprintf('2️⃣ To send just a question (no code):\n');
    fprintf('   >> ask_pulsePal_simple(''What is T1 relaxation?'')\n');
    fprintf('   This will:\n');
    fprintf('   - Copy your question to clipboard\n');
    fprintf('   - Guide you to paste in your EXISTING browser session\n\n');
    
    fprintf('3️⃣ To change settings:\n');
    fprintf('   >> edit pulsePal_config\n');
    fprintf('   You can change:\n');
    fprintf('   - use_local (true/false)\n');
    fprintf('   - session_mode (''existing'' or ''new'')\n');
    fprintf('   - auto_open_browser (true/false)\n\n');
    
    fprintf('📝 EXAMPLE SESSION:\n');
    fprintf('   1. First, open PulsePal in browser and log in (if needed)\n');
    fprintf('   2. Open your Pulseq script in MATLAB editor\n');
    fprintf('   3. Run: ask_pulsePal(''Help me debug this sequence'')\n');
    fprintf('   4. Switch to browser tab and paste (Ctrl+V)\n');
    fprintf('   5. Get specific debugging help for your code!\n');
    fprintf('   6. Continue conversation: ask_pulsePal(''Now optimize the TE'')\n');
    fprintf('   7. Paste again - context is preserved!\n\n');
    
    fprintf('💡 TIPS:\n');
    fprintf('   - Keep your browser session open for continuous work\n');
    fprintf('   - Your conversation history is preserved across queries\n');
    fprintf('   - Make sure your code is open in MATLAB editor first\n');
    fprintf('   - Keep code under 1MB for best performance\n');
    fprintf('   - You can also drag-drop .m files directly in the web interface\n\n');
    
    if strcmp(choice, '1')
        fprintf('🌐 Current URL: http://localhost:8000\n');
        fprintf('   Remember to start Chainlit first: chainlit run chainlit_app.py\n\n');
    else
        fprintf('🌐 Current URL: https://pulsepal.up.railway.app\n\n');
    end
    
    % Quick test prompt
    response = input('Would you like to test the installation now? (y/n): ', 's');
    if strcmpi(response, 'y')
        fprintf('\nRunning test with simple query...\n');
        ask_pulsePal_simple('Hello PulsePal, this is a test from MATLAB!');
    end
    
    fprintf('\n🎉 Happy MRI sequence programming with PulsePal!\n\n');
end