"""Command line entry points:  python -m futspread <cmd>"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import store
from .codes import parse_combo, parse_safe_code
from .condense import condense, load_combo, stats, summary
from .instances import (build_combo, enumerate_specs, liquid_months,
                        liquidity_table, load_contracts, month_liquidity,
                        traded_months)
from .universe import contrcode_for, universe


def _p_window(s):
    """'-40,-15' -> (-40, -15);  'all' -> None, resolved from the data."""
    if s.strip().lower() in ('all', 'full', ''):
        return None
    lo, hi = s.split(',')
    return (int(lo), int(hi))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='futspread')
    ap.add_argument('--root', default=None, help='data root (default $FUTSPREAD_ROOT)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('universe', help='list the curated universe')

    p = sub.add_parser('download', help='pull raw contracts + bars from WRDS')
    p.add_argument('--username', required=True)
    p.add_argument('--tickers', nargs='*')
    p.add_argument('--sectors', nargs='*')
    p.add_argument('--min-ltd', default='2000-01-01')

    p = sub.add_parser('describe',
                       help='ticker -> description, liquid vs illiquid months')
    p.add_argument('tickers', nargs='*')
    p.add_argument('--min-share', type=float, default=0.05)
    p.add_argument('--refresh', action='store_true',
                   help='recompute the cached liquidity scan')
    p.add_argument('--liquid-only', action='store_true',
                   help='hide series with no liquid months')
    p.add_argument('--sort', default='ticker',
                   choices=['ticker', 'sector'])

    p = sub.add_parser('catalog',
                       help='every contract series in trdstrm, not just the '
                            'curated universe (queries WRDS)')
    p.add_argument('--username', required=True)
    p.add_argument('--search', nargs='*',
                   help='match contrname or ticker, e.g. --search WHEAT COCOA')
    p.add_argument('--min-contracts', type=int, default=50)
    p.add_argument('--active-since', default=None,
                   help='only series still listing past this date')
    p.add_argument('--prefixes', nargs='*',
                   help='dsmnem exchange prefix, e.g. N C I for US venues')
    p.add_argument('--out', default=None, help='write CSV here')
    p.add_argument('--limit', type=int, default=80)

    p = sub.add_parser('months', help='months a series lists, and which trade')
    p.add_argument('ticker')

    p = sub.add_parser('liquidity',
                       help='per contract-month open interest, to see which '
                            'months actually trade')
    p.add_argument('ticker')
    p.add_argument('--metric', default='oi', choices=['oi', 'volume'])
    p.add_argument('--since-year', type=int, default=2015)

    p = sub.add_parser('build', help='build combo instances')
    p.add_argument('combos', nargs='*', help='e.g. GC[X][Z1] NG[J][X0][Z0]')
    p.add_argument('--ticker', help='build every combo for this ticker')
    p.add_argument('--all', action='store_true',
                   help='build for every downloaded series in the universe')
    p.add_argument('--dry-run', action='store_true',
                   help='count what would be built, build nothing')
    p.add_argument('--kind', default='spread', choices=['spread', 'butterfly'])
    p.add_argument('--max-offset', type=int, default=2)
    p.add_argument('--consecutive-only', action='store_true')
    p.add_argument('--all-months', action='store_true',
                   help='enumerate every LISTED month, not just liquid ones')
    p.add_argument('--min-share', type=float, default=0.05,
                   help='liquidity cutoff vs the busiest month (default 0.05)')
    p.add_argument('--lookback-years', type=int, default=2)
    p.add_argument('--min-year', type=int, default=None)

    p = sub.add_parser('prune',
                       help='delete built combos whose legs are not liquid '
                            'months (dry run unless --yes)')
    p.add_argument('--tickers', nargs='*')
    p.add_argument('--min-share', type=float, default=0.05)
    p.add_argument('--yes', action='store_true', help='actually delete')

    p = sub.add_parser('report', help='condensed stats for one combo')
    p.add_argument('combo')
    p.add_argument('--window', type=_p_window, default=None,
                   help="e.g. -40,-15 ; omit or 'all' for the full range")
    p.add_argument('--axis', default='td_to_ltd',
                   choices=['td_to_ltd', 'cal_to_ltd'])

    p = sub.add_parser('plot', help='overlay + condensed plot')
    p.add_argument('combo')
    p.add_argument('--window', type=_p_window, default=None,
                   help="e.g. -40,-15 ; omit or 'all' for the full range")
    p.add_argument('--axis', default='td_to_ltd',
                   choices=['td_to_ltd', 'cal_to_ltd'])
    p.add_argument('--outfile', default=None)
    p.add_argument('--together', action='store_true',
                   help='draw the average on the overlay instead of below it')

    a = ap.parse_args(argv)

    if a.cmd == 'universe':
        print(universe().to_string(index=False))

    elif a.cmd == 'download':
        from .download import download
        download(a.username, a.root, tickers=a.tickers, sectors=a.sectors,
                 min_ltd=a.min_ltd)

    elif a.cmd == 'catalog':
        from .download import catalog
        df = catalog(a.username, a.search, a.min_contracts, a.active_since,
                     a.prefixes)
        print(f'{len(df):,} series matched')
        if a.out:
            df.to_csv(a.out, index=False)
            print('wrote', a.out)
        with pd.option_context('display.width', 250,
                               'display.max_colwidth', 58,
                               'display.max_rows', 400):
            print(df.head(a.limit).to_string(index=False))

    elif a.cmd == 'describe':
        info = store.read(store.contracts_path(a.root))
        span_path = store.series_path(a.root)
        span = (store.read(span_path).set_index('contrcode')
                if span_path.exists() else pd.DataFrame())
        uni = universe(tickers=a.tickers or None)
        liq = liquidity_table(list(uni['ticker']), a.root, a.min_share,
                              refresh=a.refresh).set_index('ticker')
        rows = []
        for r in uni.itertuples():
            sub = info[info['contrcode'] == r.contrcode]
            if sub.empty:
                continue
            lq = liq.loc[r.ticker] if r.ticker in liq.index else None
            latest = sub.sort_values('lasttrddate').iloc[-1]
            # full WRDS span if the download recorded it, else the local
            # (min-ltd-truncated) contracts file
            first = (span.loc[r.contrcode, 'first_ltd']
                     if r.contrcode in span.index else sub['lasttrddate'].min())
            rows.append((
                r.ticker, r.sector, r.exchange,
                str(latest.get('contrname', ''))[:38],
                f'{r.lot_size:,} {r.unit}',
                lq['liquid'] if lq is not None else '?',
                f'{first:%Y-%m}',
                f"{sub['lasttrddate'].max():%Y-%m}"))
        out = pd.DataFrame(rows, columns=[
            'ticker', 'sector', 'exch', 'description', 'lot_size', 'liquid',
            'first_contract', 'last_ltd'])
        if a.liquid_only:
            out = out[out['liquid'].str.len() > 0]
        out = out.sort_values(a.sort)
        with pd.option_context('display.width', 250,
                               'display.max_colwidth', 40,
                               'display.max_rows', 200):
            print(out.to_string(index=False))

    elif a.cmd == 'liquidity':
        g = month_liquidity(a.ticker, a.root, a.since_year, a.metric)
        g = g[['code', 'n', 'med_peak', 'med_days', 'share']]
        g['liquid'] = g['share'] >= 0.05
        with pd.option_context('display.width', 200):
            print(f'{a.ticker.upper()}  median peak {a.metric} per contract, '
                  f'{a.since_year}+')
            print(g.to_string(index=False))

    elif a.cmd == 'build':
        specs = [parse_combo(c) for c in a.combos]

        tickers = []
        if a.all:
            # Only series whose bars are actually on disk.
            tickers = [tk for tk in universe()['ticker']
                       if store.prices_path(contrcode_for(tk), a.root).exists()]
            missing = len(universe()) - len(tickers)
            if missing:
                print(f'note: {missing} series in the universe have no '
                      f'downloaded bars and are skipped')
        elif a.ticker:
            tickers = [a.ticker.upper()]

        per_ticker = {}
        for tk in tickers:
            try:
                if a.all_months:
                    months = traded_months(load_contracts(tk, a.root))
                else:
                    months = liquid_months(tk, a.root, a.min_share)
                got = enumerate_specs(tk, months, a.kind,
                                      a.max_offset, a.consecutive_only)
            except Exception as exc:
                print(f'  {tk}: skipped ({exc})')
                continue
            per_ticker[tk] = (len(got), ' '.join(months))
            specs += got

        if not specs:
            ap.error('give combo codes, --ticker, or --all')

        if a.dry_run:
            for tk, (n, months) in sorted(per_ticker.items()):
                print(f'  {tk:<5s} {n:>6,} {a.kind}s   {months}')
            print(f'total {len(specs):,} {a.kind}s across '
                  f'{len(per_ticker)} series - nothing built (--dry-run)')
            return 0

        print(f'using {"all listed" if a.all_months else "liquid"} months'
              f' | building {len(specs):,} combo(s)')
        built = failed = 0
        for i, spec in enumerate(specs, start=1):
            try:
                df = build_combo(spec, a.root,
                                 lookback_years=a.lookback_years,
                                 min_year=a.min_year)
            except Exception as exc:
                failed += 1
                print(f'  {spec.code:<22s} FAILED {exc}')
                continue
            if df.empty:
                continue
            built += 1
            if len(specs) <= 40 or built % 50 == 0:
                print(f'  [{i}/{len(specs)}] {spec.code:<22s} '
                      f'{df["instance"].nunique():>3d} instances '
                      f'{len(df):>7,} rows')
        print(f'{built:,}/{len(specs):,} combos had usable instances'
              + (f', {failed} failed' if failed else ''))

    elif a.cmd == 'prune':
        base = store.root_path(a.root) / 'combos'
        want = a.tickers and [x.upper() for x in a.tickers]
        doomed, kept, liq = [], 0, {}
        for f in sorted(base.rglob('*.parquet')):
            try:
                spec = parse_safe_code(f.stem)
            except Exception:
                continue
            if want and spec.ticker not in want:
                continue
            if spec.ticker not in liq:
                try:
                    liq[spec.ticker] = set(liquid_months(spec.ticker, a.root,
                                                        a.min_share))
                except Exception:
                    liq[spec.ticker] = None
            ok = liq[spec.ticker]
            if ok is None:
                continue
            if any(lg.month not in ok for lg in spec.legs):
                doomed.append(f)
            else:
                kept += 1
        print(f'{kept:,} combos use only liquid months; '
              f'{len(doomed):,} do not')
        for f in doomed[:15]:
            print('   ', f.stem)
        if len(doomed) > 15:
            print(f'    ... and {len(doomed) - 15:,} more')
        if a.yes:
            for f in doomed:
                f.unlink()
            print(f'deleted {len(doomed):,} files')
        else:
            print('dry run - pass --yes to delete')

    elif a.cmd in ('report', 'plot'):
        from .condense import full_window
        df = load_combo(a.combo, a.root)
        try:
            spec = parse_combo(a.combo)
            ok = set(liquid_months(spec.ticker, a.root))
            thin = [lg.month for lg in spec.legs if lg.month not in ok]
            if thin:
                print(f'WARNING: leg month(s) {" ".join(thin)} are not liquid '
                      f'in {spec.ticker} (liquid: {" ".join(sorted(ok))}). '
                      f'This spread is unlikely to be tradeable.')
        except Exception:
            pass
        if a.window is None:
            a.window = full_window(df, a.axis)
            print(f'window not given - using the full range {a.window}')
        cond, wide = condense(df, a.window, a.axis)
        per = summary(df, a.window, a.axis)
        st = stats(per)
        print(f'{a.combo}  window {a.window} on {a.axis}')
        print(f'  instances: {st.get("n", 0)}   '
              f'mean {st.get("mean", float("nan")):.4f}   '
              f'median {st.get("median", float("nan")):.4f}   '
              f'hit {st.get("hit_rate", float("nan")):.1%}   '
              f't {st.get("t_stat", float("nan")):.2f}')
        print(per.to_string(index=False))
        if a.cmd == 'plot':
            import matplotlib
            if a.outfile:
                matplotlib.use('Agg')
            from .plotting import plot_combo
            fig, _ = plot_combo(df, a.window, a.axis, separate=not a.together,
                                outfile=a.outfile)
            if a.outfile:
                print('wrote', a.outfile)
            else:
                import matplotlib.pyplot as plt
                plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
