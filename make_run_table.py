#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 23 9:04 AM 2026
Created in PyCharm
Created as nTof_x17/make_run_table.py

@author: Dylan Neff, dylan
"""

import os
import json
import pandas as pd


def main():
    # run_dir = '/media/dylan/data/x17/feb_beam/runs/'
    run_dir = '/media/dylan/data/x17/feb_beam/runs/'
    run_cfg_name = 'run_config.json'
    eos_site_dir = '/eos/user/d/dneff/www/'

    trigger_types = {
        'Tcm_Mx17_Feb_test.cfg': 'PS',
        'Self_Tcm_MM_Mx17_Feb_test.cfg': 'Self'
    }

    df = []
    for run in os.listdir(run_dir):
        run_config_path = os.path.join(run_dir, run, run_cfg_name)
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        if run_config['run_name'] != run:
            print(f'Run name in {run_cfg_name} does not match directory name: {run_config["run_name"]} != {run}')

        trig_type = os.path.basename(run_config['dream_daq_info']['daq_config_template_path'])
        trig_type = trigger_types.get(trig_type)
        if trig_type is None:
            trig_type = 'Unknown'

        sub_run_dirs = [x for x in os.listdir(os.path.join(run_dir, run))]

        sub_runs = run_config['sub_runs']


        run_row = {
            'run': int(run_config['run_name'].split('_')[-1]),
            'start_time': run_config['start_time'],
            'gas': run_config['gas'],
            'beam_type': run_config['beam_type'],
            'target_type': run_config['target_type'],
            'trigger_type': trig_type
        }

        df.append(run_row)
    df = pd.DataFrame(df)

    gen_site(df, eos_site_dir)

    print('donzo')


def gen_site(df, site_dir):
    """
    Generate site on eos
    """
    os.makedirs(site_dir, exist_ok=True)

    # --- 2. Shared HTML Components ---
    # Using Simple.css for instant styling
    HTML_HEAD = """<head>
        <meta charset="UTF-8">
        <title>CERN Beam Data</title>
        <link rel="stylesheet" href="https://cdn.simplecss.org/simple.min.css">
    </head>"""

    # --- 3. Generate Index Page (The Table) ---
    index_content = f"""
    <html>
    {HTML_HEAD}
    <body>
        <header><h1>Beam Run Registry</h1></header>
        <main>
        <table>
            <thead>
                <tr>
                    <th>Run #</th>
                    <th>Start Time</th>
                    <th>Beam</th>
                    <th>Target</th>
                </tr>
            </thead>
            <tbody>
    """

    for run in df.to_dict(orient='records'):
        # Add row to index table
        index_content += f"""
            <tr>
                <td><a href="run_{run['run']}.html"><b>{run['run']}</b></a></td>
                <td>{run['start_time']}</td>
                <td>{run['beam_type']}</td>
                <td>{run['target_type']}</td>
            </tr>
        """

        # --- 4. Generate Detail Page for this run ---
        detail_content = f"""
        <html>
        {HTML_HEAD}
        <body>
            <header>
                <nav><a href="index.html">← Back to List</a></nav>
                <h1>Run {run['run']}</h1>
            </header>
            <main>
                <article>
                    <h3>Configuration Details</h3>
                    <ul>
                        <li><strong>Start Time:</strong> {run['start_time']}</li>
                        <li><strong>Gas Mixture:</strong> {run['gas']}</li>
                        <li><strong>Beam Type:</strong> {run['beam_type']}</li>
                        <li><strong>Target:</strong> {run['target_type']}</li>
                        <li><strong>Trigger:</strong> {run['trigger_type']}</li>
                    </ul>
                </article>
                <section>
                    <h4>Plots & Analysis</h4>
                    <p><i>If you have run_{run['run']}.png in this folder, it would show up here:</i></p>
                    <img src="plots/run_{run['run']}.png" alt="Run plot" style="max-width:100%; border:1px solid #ccc;">
                </section>
            </main>
        </body>
        </html>
        """

        with open(f"{site_dir}/run_{run['run']}.html", "w") as f:
            f.write(detail_content)

    index_content += "</tbody></table></body></html>"
    with open(f"{site_dir}/index.html", "w") as f:
        f.write(index_content)

    print(f"Site generated in {site_dir}/")


if __name__ == '__main__':
    main()
