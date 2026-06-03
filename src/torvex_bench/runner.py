"""
runner.py is the orchestrator.

It loads samples from the runtime manifest.

For each sample, it sends sample.pdf_path to the selected adapter.

The adapter runs the engine and returns DocumentResult.

The runner passes the DocumentResult and sample ground truth to table_structure.py.

Then it saves per-sample scores and summary results.

"""