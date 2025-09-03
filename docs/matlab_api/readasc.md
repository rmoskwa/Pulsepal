# readasc

Reads Siemens ASC ASCII-formatted text files and returns a MATLAB structure containing the data.  Handles files that may be split into two parts (e.g., ####.asc and ####_GSWD_SAFETY.asc), merging the data from both if found.

## Syntax

```matlab
function [asc, extra] = readasc(filePathBasic)
```

## Calling Pattern

```matlab
mr.Siemens.readasc(...)
```

## Parameters

### Required Parameters

| Name | Type | Description | Example | Units |
|------|------|-------------|---------|-------|
| `filePathBasic` | string | The base path and filename of the Siemens ASC file (e.g., 'path/to/file.asc'). The function will automatically check for a corresponding '_GSWD_SAFETY' file if it exists. | `'path/to/my_scan.asc'` |  |

## Returns

| Output | Type | Description |
|--------|------|-------------|
| `asc` | struct | A MATLAB structure containing the data read from the ASC file(s). Field names and values correspond to the parameters defined in the ASC file. |
| `extra` | struct | This output is currently not used by the provided code excerpt.  It might be used for additional extracted information in the full function implementation. |

## Examples

```matlab
myAsc = mr.Siemens.readasc('path/to/my_scan.asc');
[prot, yaps] = mr.Siemens.readasc('path/to/another_scan.asc');
```
