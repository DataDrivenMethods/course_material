import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import calendar
import warnings
warnings.filterwarnings('ignore')

class FuturesDataDownloader:
    """
    A comprehensive US futures data downloader for individual monthly contracts.
    Supports CME Group (CME, CBOT, NYMEX, COMEX) and ICE futures.
    Downloads individual contracts rather than rolled/continuous series.
    """
    
    def __init__(self, databento_api_key: Optional[str] = None):
        """
        Initialize the futures data downloader.
        
        Args:
            databento_api_key: Optional API key for Databento (premium service)
        """
        self.databento_api_key = databento_api_key
        
        # Month codes for futures contracts
        self.month_codes = {
            1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
            7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
        }
        
        # Reverse mapping
        self.code_to_month = {v: k for k, v in self.month_codes.items()}
        
        # Exchange suffix mapping for Yahoo Finance
        self.exchange_suffixes = {
            'NYMEX': '.NYM',
            'COMEX': '.CMX', 
            'CBOT': '.CBT',
            'ICE': '.ICE',
            'CME': '.CME'
        }
        
        # Major US futures contracts by category
        self.futures_contracts = {
            # Energy (NYMEX)
            'energy': {
                'CL': {'name': 'Crude Oil WTI', 'exchange': 'NYMEX', 'tick_size': 0.01, 'yahoo_symbol': 'CL', 'suffix': '.NYM'},
                'NG': {'name': 'Natural Gas', 'exchange': 'NYMEX', 'tick_size': 0.001, 'yahoo_symbol': 'NG', 'suffix': '.NYM'},
                'HO': {'name': 'Heating Oil', 'exchange': 'NYMEX', 'tick_size': 0.0001, 'yahoo_symbol': 'HO', 'suffix': '.NYM'},
                'RB': {'name': 'RBOB Gasoline', 'exchange': 'NYMEX', 'tick_size': 0.0001, 'yahoo_symbol': 'RB', 'suffix': '.NYM'},
                'BZ': {'name': 'Brent Crude Oil', 'exchange': 'NYMEX', 'tick_size': 0.01, 'yahoo_symbol': 'BZ', 'suffix': '.NYM'},
            },
            
            # Agriculture (CBOT)
            'agriculture': {
                'ZC': {'name': 'Corn', 'exchange': 'CBOT', 'tick_size': 0.25, 'yahoo_symbol': 'ZC', 'suffix': '.CBT'},
                'ZS': {'name': 'Soybeans', 'exchange': 'CBOT', 'tick_size': 0.25, 'yahoo_symbol': 'ZS', 'suffix': '.CBT'},
                'ZW': {'name': 'Soft Red Winter Wheat', 'exchange': 'CBOT', 'tick_size': 0.25, 'yahoo_symbol': 'ZW', 'suffix': '.CBT'},
                'ZM': {'name': 'Soybean Meal', 'exchange': 'CBOT', 'tick_size': 0.10, 'yahoo_symbol': 'ZM', 'suffix': '.CBT'},
                'ZL': {'name': 'Soybean Oil', 'exchange': 'CBOT', 'tick_size': 0.01, 'yahoo_symbol': 'ZL', 'suffix': '.CBT'},
                'ZO': {'name': 'Oats', 'exchange': 'CBOT', 'tick_size': 0.25, 'yahoo_symbol': 'ZO', 'suffix': '.CBT'},
                'KE': {'name': 'Kansas City Hard Red Winter Wheat', 'exchange': 'CBOT', 'tick_size': 0.25, 'yahoo_symbol': 'KE', 'suffix': '.CBT'},
            },
            
            # Softs (ICE/CBOT)
            'softs': {
                'CT': {'name': 'Cotton', 'exchange': 'ICE', 'tick_size': 0.01, 'yahoo_symbol': 'CT', 'suffix': '.ICE'},
                'CC': {'name': 'Cocoa', 'exchange': 'ICE', 'tick_size': 1.0, 'yahoo_symbol': 'CC', 'suffix': '.ICE'},
                'KC': {'name': 'Coffee', 'exchange': 'ICE', 'tick_size': 0.05, 'yahoo_symbol': 'KC', 'suffix': '.ICE'},
                'SB': {'name': 'Sugar #11', 'exchange': 'ICE', 'tick_size': 0.01, 'yahoo_symbol': 'SB', 'suffix': '.ICE'},
                'OJ': {'name': 'Orange Juice', 'exchange': 'ICE', 'tick_size': 0.05, 'yahoo_symbol': 'OJ', 'suffix': '.ICE'},
            },
            
            # Metals (COMEX)
            'metals': {
                'GC': {'name': 'Gold', 'exchange': 'COMEX', 'tick_size': 0.10, 'yahoo_symbol': 'GC', 'suffix': '.CMX'},
                'SI': {'name': 'Silver', 'exchange': 'COMEX', 'tick_size': 0.005, 'yahoo_symbol': 'SI', 'suffix': '.CMX'},
                'HG': {'name': 'High Grade Copper', 'exchange': 'COMEX', 'tick_size': 0.0005, 'yahoo_symbol': 'HG', 'suffix': '.CMX'},
                'PL': {'name': 'Platinum', 'exchange': 'COMEX', 'tick_size': 0.10, 'yahoo_symbol': 'PL', 'suffix': '.CMX'},
                'PA': {'name': 'Palladium', 'exchange': 'COMEX', 'tick_size': 0.05, 'yahoo_symbol': 'PA', 'suffix': '.CMX'},
            }
        }
        
        # Contract months by product (some contracts don't trade all months)
        self.contract_months = {
            # Energy - typically trade all months
            'CL': [1,2,3,4,5,6,7,8,9,10,11,12],  # All months
            'NG': [1,2,3,4,5,6,7,8,9,10,11,12],  # All months
            'HO': [1,2,3,4,5,6,7,8,9,10,11,12],  # All months
            'RB': [1,2,3,4,5,6,7,8,9,10,11,12],  # All months
            'BZ': [1,2,3,4,5,6,7,8,9,10,11,12],  # All months
            
            # Agriculture - specific months
            'ZC': [3,5,7,9,12],  # March, May, July, September, December
            'ZS': [1,3,5,7,8,9,11],  # Jan, Mar, May, Jul, Aug, Sep, Nov
            'ZW': [3,5,7,9,12],  # March, May, July, September, December
            'ZM': [1,3,5,7,8,9,10,12],  # Jan, Mar, May, Jul, Aug, Sep, Oct, Dec
            'ZL': [1,3,5,7,8,9,10,12],  # Jan, Mar, May, Jul, Aug, Sep, Oct, Dec
            'ZO': [3,5,7,9,12],  # March, May, July, September, December
            'KE': [3,5,7,9,12],  # March, May, July, September, December
            
            # Softs
            'CT': [3,5,7,10,12],  # March, May, July, October, December
            'CC': [3,5,7,9,12],  # March, May, July, September, December
            'KC': [3,5,7,9,12],  # March, May, July, September, December
            'SB': [3,5,7,10],  # March, May, July, October
            'OJ': [1,3,5,7,9,11],  # Jan, Mar, May, Jul, Sep, Nov
            
            # Metals
            'GC': [2,4,6,8,10,12],  # Feb, Apr, Jun, Aug, Oct, Dec
            'SI': [3,5,7,9,12],  # March, May, July, September, December
            'HG': [3,5,7,9,12],  # March, May, July, September, December
            'PL': [1,4,7,10],  # January, April, July, October
            'PA': [1,4,7,10],  # January, April, July, October
        }
    
    def generate_contract_symbol(self, base_symbol: str, month: int, year: int, source: str = 'yahoo') -> str:
        """
        Generate the full contract symbol for a specific month and year.
        
        Args:
            base_symbol: Base symbol (e.g., 'CL', 'GC', 'ZC')
            month: Contract month (1-12)
            year: Contract year (e.g., 2023)
            source: Data source ('yahoo', 'databento')
            
        Returns:
            Full contract symbol (e.g., 'CLZ23.NYM' for Yahoo, 'CLZ23' for others)
        """
        month_code = self.month_codes.get(month, 'F')
        year_code = str(year)[-2:]  # Last 2 digits
        
        if source == 'yahoo':
            # Find the exchange suffix for this base symbol
            exchange_suffix = ''
            for category, contracts in self.futures_contracts.items():
                if base_symbol in contracts:
                    exchange_suffix = contracts[base_symbol].get('suffix', '')
                    break
            
            return f"{base_symbol}{month_code}{year_code}{exchange_suffix}"
        else:
            return f"{base_symbol}{month_code}{year_code}"
    
    def parse_contract_symbol(self, symbol: str) -> Tuple[str, int, int]:
        """
        Parse a contract symbol to extract base symbol, month, and year.
        
        Args:
            symbol: Contract symbol (e.g., 'CLZ23.NYM', 'GCM24.CMX')
            
        Returns:
            Tuple of (base_symbol, month, year)
        """
        # Remove Yahoo Finance suffix if present (both old and new formats)
        clean_symbol = symbol.replace('=F', '')
        
        # Remove exchange suffix (.NYM, .CMX, .CBT, .ICE)
        for suffix in ['.NYM', '.CMX', '.CBT', '.ICE', '.CME']:
            if clean_symbol.endswith(suffix):
                clean_symbol = clean_symbol[:-len(suffix)]
                break
        
        if len(clean_symbol) >= 4:
            # Extract components - most symbols are like CLZ23, GCM24
            if len(clean_symbol) == 4:  # Like GCM4 (single digit year)
                base_symbol = clean_symbol[:2]
                month_code = clean_symbol[2]
                year_code = clean_symbol[3]
            elif len(clean_symbol) == 5:  # Like CLZ23, GCM24
                base_symbol = clean_symbol[:2]
                month_code = clean_symbol[2]
                year_code = clean_symbol[3:5]
            else:  # Longer symbols
                base_symbol = clean_symbol[:-3]
                month_code = clean_symbol[-3]
                year_code = clean_symbol[-2:]
            
            # Convert month code to number
            month = self.code_to_month.get(month_code, 1)
            
            # Convert year code to full year
            year_int = int(year_code)
            if year_int < 50:  # Assume 2000s
                year = 2000 + year_int
            else:  # Assume 1900s
                year = 1900 + year_int
            
            return base_symbol, month, year
        
        return symbol, 1, 2023
    
    def get_contract_expiry_date(self, base_symbol: str, month: int, year: int) -> datetime:
        """
        Estimate contract expiry date based on typical patterns.
        Note: These are approximations - actual expiry dates may vary.
        
        Args:
            base_symbol: Base symbol
            month: Contract month
            year: Contract year
            
        Returns:
            Estimated expiry date
        """
        # Most futures expire on the third Friday of the contract month
        # Energy futures often expire on the third business day before the 25th
        
        if base_symbol in ['CL', 'NG', 'HO', 'RB', 'BZ']:  # Energy
            # Typically expire on the third business day before the 25th
            expiry_day = 22  # Approximate
        elif base_symbol in ['GC', 'SI', 'HG', 'PL', 'PA']:  # Metals
            # Typically expire on the third-to-last business day of the month
            last_day = calendar.monthrange(year, month)[1]
            expiry_day = last_day - 3
        else:  # Agriculture and Softs
            # Typically expire around the 14th of the month
            expiry_day = 14
        
        return datetime(year, month, min(expiry_day, calendar.monthrange(year, month)[1]))
    
    def generate_contract_list(self, base_symbol: str, start_year: int, end_year: int) -> List[Dict]:
        """
        Generate list of all individual contracts for a symbol over a date range.
        
        Args:
            base_symbol: Base symbol (e.g., 'CL', 'GC')
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            List of contract dictionaries with symbol info
        """
        contracts = []
        
        if base_symbol not in self.contract_months:
            print(f"Warning: Contract months not defined for {base_symbol}")
            return contracts
        
        trading_months = self.contract_months[base_symbol]
        
        for year in range(start_year, end_year + 1):
            for month in trading_months:
                contract_symbol = self.generate_contract_symbol(base_symbol, month, year, 'yahoo')
                expiry_date = self.get_contract_expiry_date(base_symbol, month, year)
                
                # Find contract info
                contract_info = None
                for category, contracts_dict in self.futures_contracts.items():
                    if base_symbol in contracts_dict:
                        contract_info = contracts_dict[base_symbol]
                        break
                
                contract_data = {
                    'symbol': contract_symbol,
                    'base_symbol': base_symbol,
                    'month': month,
                    'year': year,
                    'month_code': self.month_codes[month],
                    'expiry_date': expiry_date,
                    'name': contract_info['name'] if contract_info else base_symbol,
                    'exchange': contract_info['exchange'] if contract_info else 'Unknown',
                    'category': self._get_contract_category(base_symbol)
                }
                
                contracts.append(contract_data)
        
        return contracts
    
    def _get_contract_category(self, base_symbol: str) -> str:
        """Get the category for a base symbol."""
        for category, contracts_dict in self.futures_contracts.items():
            if base_symbol in contracts_dict:
                return category
        return 'unknown'
    
    def download_contract_data(self, contract_symbol: str, start_date: Optional[str] = None, 
                              end_date: Optional[str] = None, fallback_to_continuous: bool = True) -> pd.DataFrame:
        """
        Download historical data for a specific futures contract.
        
        Args:
            contract_symbol: Full contract symbol (e.g., 'CLZ23.NYM')
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            fallback_to_continuous: Try continuous contract if individual contract fails
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(contract_symbol)
            
            # If no dates specified, try to get reasonable range
            if not start_date and not end_date:
                # Try to get max data
                data = ticker.history(period='max')
            else:
                data = ticker.history(start=start_date, end=end_date)
            
            if data.empty and fallback_to_continuous:
                print(f"No data for individual contract {contract_symbol}, trying continuous contract...")
                
                # Try continuous contract as fallback
                base_symbol, month, year = self.parse_contract_symbol(contract_symbol)
                continuous_symbol = f"{base_symbol}=F"
                
                ticker = yf.Ticker(continuous_symbol)
                if not start_date and not end_date:
                    data = ticker.history(period='max')
                else:
                    data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    print(f"Successfully downloaded continuous data for {continuous_symbol}")
                    # Mark this as continuous data
                    contract_symbol = continuous_symbol + " (continuous fallback)"
            
            if data.empty:
                print(f"No data found for {contract_symbol}")
                return pd.DataFrame()
            
            # Clean and enhance the data
            data = data.reset_index()
            data['Symbol'] = contract_symbol
            
            # Parse contract info
            base_symbol, month, year = self.parse_contract_symbol(contract_symbol)
            data['Base_Symbol'] = base_symbol
            data['Contract_Month'] = month
            data['Contract_Year'] = year
            data['Month_Code'] = self.month_codes.get(month, 'F')
            
            # Add expiry date
            expiry_date = self.get_contract_expiry_date(base_symbol, month, year)
            data['Expiry_Date'] = expiry_date
            
            # Add days to expiry (handle timezone issues)
            if data['Date'].dt.tz is not None:
                expiry_date_tz = pd.to_datetime(expiry_date).tz_localize(data['Date'].dt.tz)
                data['Days_To_Expiry'] = (expiry_date_tz - data['Date']).dt.days
            else:
                data['Days_To_Expiry'] = (pd.to_datetime(expiry_date) - data['Date']).dt.days
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for {contract_symbol}: {str(e)}")
            
            # Try one more fallback - continuous contract
            if fallback_to_continuous and "=F" not in contract_symbol:
                print(f"Trying continuous contract fallback...")
                base_symbol, month, year = self.parse_contract_symbol(contract_symbol)
                return self.download_contract_data(f"{base_symbol}=F", start_date, end_date, False)
            
            return pd.DataFrame()
    
    def download_multiple_contracts(self, base_symbol: str, start_year: int, end_year: int,
                                   data_start_date: Optional[str] = None,
                                   data_end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download data for all contracts of a futures series over a year range.
        
        Args:
            base_symbol: Base symbol (e.g., 'CL', 'GC')
            start_year: Starting year for contracts
            end_year: Ending year for contracts
            data_start_date: Start date for historical data
            data_end_date: End date for historical data
            
        Returns:
            Combined DataFrame with all contract data
        """
        print(f"Generating contract list for {base_symbol} ({start_year}-{end_year})...")
        contracts = self.generate_contract_list(base_symbol, start_year, end_year)
        
        all_data = []
        successful_downloads = 0
        
        for contract in contracts:
            contract_symbol = contract['symbol']
            print(f"Downloading {contract_symbol} (expires: {contract['expiry_date'].strftime('%Y-%m-%d')})...")
            
            # Determine data range - limit to before expiry
            start_date = data_start_date
            end_date = data_end_date or contract['expiry_date'].strftime('%Y-%m-%d')
            
            data = self.download_contract_data(contract_symbol, start_date, end_date)
            
            if not data.empty:
                successful_downloads += 1
                all_data.append(data)
            
            # Be respectful to Yahoo Finance
            time.sleep(0.2)
        
        print(f"Successfully downloaded {successful_downloads}/{len(contracts)} contracts")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['Contract_Year', 'Contract_Month', 'Date'])
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_active_contracts_on_date(self, base_symbol: str, target_date: str) -> List[str]:
        """
        Get list of contracts that were actively trading on a specific date.
        
        Args:
            base_symbol: Base symbol
            target_date: Date in YYYY-MM-DD format
            
        Returns:
            List of contract symbols
        """
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        active_contracts = []
        
        # Look at a reasonable range around the target date
        start_year = target_dt.year - 1
        end_year = target_dt.year + 2
        
        contracts = self.generate_contract_list(base_symbol, start_year, end_year)
        
        for contract in contracts:
            # Contract is active if target date is before expiry
            # and after reasonable listing date (assume 18 months before expiry)
            listing_date = contract['expiry_date'] - timedelta(days=540)  # 18 months
            
            if listing_date <= target_dt <= contract['expiry_date']:
                active_contracts.append(contract['symbol'])
        
        return active_contracts
    
    def analyze_contract_curve(self, data: pd.DataFrame, analysis_date: str) -> pd.DataFrame:
        """
        Analyze the futures curve for a specific date.
        
        Args:
            data: Combined futures data
            analysis_date: Date to analyze in YYYY-MM-DD format
            
        Returns:
            DataFrame with curve analysis
        """
        analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d').date()
        
        # Filter data for the analysis date
        date_data = data[data['Date'].dt.date == analysis_dt].copy()
        
        if date_data.empty:
            print(f"No data found for {analysis_date}")
            return pd.DataFrame()
        
        # Calculate curve metrics
        date_data = date_data.sort_values('Days_To_Expiry')
        date_data['Price_Change'] = date_data['Close'].pct_change()
        date_data['Contango_Backwardation'] = date_data['Close'].diff()
        
        # Mark front month (shortest days to expiry)
        date_data['Is_Front_Month'] = date_data['Days_To_Expiry'] == date_data['Days_To_Expiry'].min()
        
        return date_data[['Symbol', 'Close', 'Volume', 'Days_To_Expiry', 
                         'Contract_Month', 'Contract_Year', 'Price_Change',
                         'Contango_Backwardation', 'Is_Front_Month']]

    def check_data_availability(self, base_symbol: str, test_contracts: int = 3) -> Dict[str, Any]:
        """
        Check data availability for a futures symbol.
        
        Args:
            base_symbol: Base symbol to test
            test_contracts: Number of recent contracts to test
            
        Returns:
            Dictionary with availability information
        """
        current_year = datetime.now().year
        results = {
            'base_symbol': base_symbol,
            'individual_contracts_available': False,
            'continuous_contract_available': False,
            'working_contracts': [],
            'failed_contracts': [],
            'recommendations': []
        }
        
        # Test continuous contract first
        continuous_symbol = f"{base_symbol}=F"
        try:
            ticker = yf.Ticker(continuous_symbol)
            data = ticker.history(period='5d')  # Small test
            if not data.empty:
                results['continuous_contract_available'] = True
                print(f"✓ Continuous contract {continuous_symbol} available")
        except:
            pass
        
        # Test individual contracts
        if base_symbol in self.contract_months:
            trading_months = self.contract_months[base_symbol][:3]  # Test first 3 months
            
            for month in trading_months:
                contract_symbol = self.generate_contract_symbol(base_symbol, month, current_year)
                
                try:
                    ticker = yf.Ticker(contract_symbol)
                    data = ticker.history(period='5d')
                    
                    if not data.empty:
                        results['working_contracts'].append(contract_symbol)
                        results['individual_contracts_available'] = True
                        print(f"✓ Individual contract {contract_symbol} available")
                    else:
                        results['failed_contracts'].append(contract_symbol)
                        print(f"✗ Individual contract {contract_symbol} not available")
                except Exception as e:
                    results['failed_contracts'].append(f"{contract_symbol} (error: {str(e)[:50]})")
                    print(f"✗ Individual contract {contract_symbol} failed: {str(e)[:50]}")
        
        # Generate recommendations
        if results['individual_contracts_available']:
            results['recommendations'].append("Individual contracts are available - use download_contract_data()")
        elif results['continuous_contract_available']:
            results['recommendations'].append("Only continuous contracts available - individual contracts not supported for this symbol")
            results['recommendations'].append("Consider using continuous data or alternative data sources")
        else:
            results['recommendations'].append("No data available via Yahoo Finance - try alternative sources")
            results['recommendations'].append("Consider premium APIs like Databento, CME DataMine, or Quandl")
        
        return results

# Example usage and helper functions
def demo():
    print("=== US Futures Data Download Demo (Fixed Version) ===\n")
    
    # Initialize downloader
    downloader = FuturesDataDownloader()
    
    print("1. Data Availability Check")
    print("-" * 40)
    print("Checking what futures data is actually available via Yahoo Finance...\n")
    
    # Test key symbols for availability
    test_symbols = ['CL', 'RB', 'HO', 'NG']
    availability_results = {}
    
    for symbol in test_symbols:
        print(f"Testing {symbol}...")
        results = downloader.check_data_availability(symbol)
        availability_results[symbol] = results
        
        if results['individual_contracts_available']:
            print(f"  ✓ {symbol}: Individual contracts work!")
        elif results['continuous_contract_available']:
            print(f"  ⚠ {symbol}: Only continuous contracts available")
        else:
            print(f"  ✗ {symbol}: No data available")
        
        print()
    
    print("="*60)
    print("2. Working Examples Based on Availability")
    print("-" * 40)
    
    # Find symbols that actually work
    working_individual = [s for s, r in availability_results.items() if r['individual_contracts_available']]
    working_continuous = [s for s, r in availability_results.items() if r['continuous_contract_available'] and not r['individual_contracts_available']]
    
    if working_individual:
        print(f"✓ Symbols with individual contracts: {', '.join(working_individual)}")
        
        # Demo with first working symbol
        demo_symbol = working_individual[0]
        print(f"\nTesting individual contracts for {demo_symbol}...")
        
        # Get a recent contract
        contracts = downloader.generate_contract_list(demo_symbol, 2024, 2024)
        if contracts:
            contract = contracts[0]
            print(f"Downloading {contract['symbol']}...")
            
            data = downloader.download_contract_data(contract['symbol'])
            if not data.empty:
                print(f"✓ Successfully downloaded {len(data)} rows")
                print("Sample data:")
                print(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
                
                # Save the data
                filename = f"{contract['symbol'].replace('.', '_')}_data.csv"
                data.to_csv(filename, index=False)
                print(f"✓ Saved to {filename}")
            else:
                print("✗ No data retrieved")
    else:
        print("✗ No symbols support individual contracts")
    
    print("\n" + "="*60)
    
    if working_continuous:
        print(f"✓ Symbols with continuous contracts: {', '.join(working_continuous)}")
        
        # Demo continuous contract
        demo_symbol = working_continuous[0]
        print(f"\nDownloading continuous contract for {demo_symbol}...")
        
        try:
            continuous_data = downloader.download_contract_data(f"{demo_symbol}=F", fallback_to_continuous=False)
        except Exception as e:
            print(f"Error downloading {demo_symbol}=F: {e}")
            continuous_data = pd.DataFrame()
        
        if not continuous_data.empty:
            print(f"✓ Downloaded {len(continuous_data)} rows of continuous data")
            print("Recent prices:")
            print(continuous_data[['Date', 'Close', 'Volume']].tail())
            
            # Save continuous data
            filename = f"{demo_symbol}_continuous_data.csv"
            continuous_data.to_csv(filename, index=False)
            print(f"✓ Saved to {filename}")
    else:
        print("✗ No symbols support continuous contracts")
    
    print("\n" + "="*60)
    print("4. Contract Specifications Reference")
    print("-" * 40)
    
    print("US Futures Contract Details:")
    for category, contracts in downloader.futures_contracts.items():
        print(f"\n{category.upper()}:")
        for symbol, info in contracts.items():
            months = downloader.contract_months.get(symbol, [])
            month_names = [calendar.month_name[m][:3] for m in months]
            availability = "✓" if symbol in working_individual else "⚠" if symbol in working_continuous else "?"
            print(f"  {availability} {symbol:3} - {info['name']:<25} ({info['exchange']}) - {', '.join(month_names)}")
    
    print(f"\n✓ = Individual contracts available")
    print(f"⚠ = Only continuous contracts available") 
    # print(f"✗ = No data available via Yahoo Finance")
    print(f"?  = Untested")    
    
    print("\n" + "="*60)
    print("5. Summary and Next Steps")
    print("-" * 30)
    
    total_symbols = len(test_symbols)
    working_individual_count = len(working_individual)
    working_continuous_count = len(working_continuous)
    
    print(f"Data Availability Summary:")
    print(f"• Total symbols tested: {total_symbols}")
    print(f"• Individual contracts working: {working_individual_count}")
    print(f"• Continuous contracts only: {working_continuous_count}")
    print(f"• No data available: {total_symbols - working_individual_count - working_continuous_count}")
    
    print(f"\nRecommendations:")
    if working_individual_count > 0:
        print(f"• Use the working symbols ({', '.join(working_individual)}) for development")
        print(f"• Implement your roll strategies with available data")
    
    if working_continuous_count > 0:
        print(f"• Use continuous contracts as a fallback for analysis")
    
    print(f"\nFiles created:")
    created_files = []
    
    # Download all working continuous contracts
    for symbol in working_continuous:
        try:
            print(f"Downloading {symbol}=F...")
            data = downloader.download_contract_data(f"{symbol}=F", fallback_to_continuous=False)
            if not data.empty:
                filename = f"{symbol}_continuous_data.csv"
                data.to_csv(filename, index=False)
                created_files.append(filename)
                print(f"✓ Saved {filename} ({len(data)} rows)")
            else:
                print(f"✗ No data for {symbol}=F")
        except Exception as e:
            print(f"✗ Error downloading {symbol}=F: {e}")
    
    for symbol in working_individual:
        contracts = downloader.generate_contract_list(symbol, 2024, 2024)
        if contracts:
            filename = f"{contracts[0]['symbol'].replace('.', '_')}_data.csv"
            created_files.append(filename)
    
    for file in created_files:
        print(f"• {file}")
    
    return downloader, availability_results

if __name__ == "__main__":
    downloader, results = demo()

